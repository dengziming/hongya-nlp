/*
 * Copyright 2015 and onwards Sanford Ryza, Uri Laserson, Sean Owen and Joshua Wills
 *
 * See LICENSE file for further information.
 */

package com.hongya.bigdata.nlp.analyzer

import edu.umd.cloud9.collection.XMLInputFormat
import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.umd.cloud9.collection.wikipedia.WikipediaPage
import edu.umd.cloud9.collection.wikipedia.language.EnglishWikipediaPage
import java.util.Properties

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.spark.ml.feature.{CountVectorizer, IDF}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

class AssembleDocumentTermMatrix(val spark: SparkSession) extends Serializable {
  import spark.implicits._

  /**
   * 返回 (标题,内容)
   */
  def wikiXmlToPlainText(pageXml: String): Option[(String, String)] = {
    val page = new EnglishWikipediaPage()

    // 维基百科的xml文件有时候会发生变化，这样处理是为了防止有些xml文件出现异常，解析依然生效
    //
    val hackedPageXml = pageXml.replaceFirst(
      "<text xml:space=\"preserve\" bytes=\"\\d+\">", "<text xml:space=\"preserve\">")

    WikipediaPage.readPage(page, hackedPageXml)
    if (page.isEmpty || !page.isArticle || page.isRedirect || page.isDisambiguation ||
        page.getTitle.contains("(disambiguation)")) {
      None
    } else {
      Some((page.getTitle, page.getContent))
    }
  }

  def parseWikipediaDump(path: String): Dataset[(String, String)] = {
    val conf = new Configuration()
    conf.set(XMLInputFormat.START_TAG_KEY, "<page>")
    conf.set(XMLInputFormat.END_TAG_KEY, "</page>")
    val kvs = spark.sparkContext.newAPIHadoopFile(path, classOf[XMLInputFormat], classOf[LongWritable],
      classOf[Text], conf)
    val rawXmls = kvs.map(_._2.toString).toDS()

    rawXmls.filter(_ != null).flatMap(wikiXmlToPlainText)
  }

  /**
   * 创建StanfordCoreNLP pipeline 对象，将文档进行词法分析
   */
  def createNLPPipeline(): StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "tokenize, ssplit, pos, lemma")
    new StanfordCoreNLP(props)
  }

  /**
    *
    * 判断是不是一个字符
    */
  def isOnlyLetters(str: String): Boolean = {
    str.forall(c => Character.isLetter(c))
  }

  /**
    *  文本变成词法单元，需要使用 pipeline
    */
  def plainTextToLemmas(text: String, stopWords: Set[String], pipeline: StanfordCoreNLP)
    : Seq[String] = {

    val doc = new Annotation(text)
    pipeline.annotate(doc)

    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation])
    for (sentence <- sentences.asScala;
         token <- sentence.get(classOf[TokensAnnotation]).asScala) {
      val lemma = token.get(classOf[LemmaAnnotation])
      if (lemma.length > 2 && !stopWords.contains(lemma) && isOnlyLetters(lemma)) {
        lemmas += lemma.toLowerCase
      }
    }
    lemmas
  }

  /**
    * 加载无用的词
    * @param path 路径
    */
  def loadStopWords(path: String): Set[String] = {
    scala.io.Source.fromFile(path).getLines().toSet
  }

  /**
    * 得到词法单元
    */
  def contentsToTerms(docs: Dataset[(String, String)], stopWordsFile: String): Dataset[(String, Seq[String])] = {
    val stopWords = scala.io.Source.fromFile(stopWordsFile).getLines().toSet
    val bStopWords = spark.sparkContext.broadcast(stopWords)

    docs.mapPartitions { iter =>
      val pipeline = createNLPPipeline()
      iter.map { case (title, contents) => (title, plainTextToLemmas(contents, bStopWords.value, pipeline)) }
    }
  }


  /**
   * 返回文档-词法矩阵，矩阵的每个元素是文档每一行的词项的TF-IDF
   *
   * @param docTexts a DF with two columns: title and text
   */
  def documentTermMatrix(docTexts: Dataset[(String, String)], stopWordsFile: String, numTerms: Int)
    : (DataFrame, Array[String], Map[Long, String], Array[Double]) = {

    // 首先我们调用上面的`contentsToTerms`方法得到词法单元：
    val terms = contentsToTerms(docTexts, stopWordsFile)

    // 然后我们给它添加schema，数据分为标题和内容，并且对数据进行过滤：
    val termsDF = terms.toDF("title", "terms")
    val filtered = termsDF.where(size($"terms") > 1)


    // 然后类似第一部分切分词法单元的过程，新建一个模型，将文档切分成词法:
    val countVectorizer = new CountVectorizer()
      .setInputCol("terms").setOutputCol("termFreqs").setVocabSize(numTerms)
    val vocabModel = countVectorizer.fit(filtered)
    val docTermFreqs = vocabModel.transform(filtered)

    val termIds = vocabModel.vocabulary

    docTermFreqs.cache()

    // 并给每个词项一个单独的id：
    val docIds = docTermFreqs.rdd.map(_.getString(0)).zipWithUniqueId().map(_.swap).collect().toMap

    // 然后利用spark自带的IDF模型，计算文档的TF-IDF
    val idf = new IDF().setInputCol("termFreqs").setOutputCol("tfidfVec")
    val idfModel = idf.fit(docTermFreqs)
    val docTermMatrix = idfModel.transform(docTermFreqs).select("title", "tfidfVec")

    (docTermMatrix, termIds, docIds, idfModel.idf.toArray)
  }
}
