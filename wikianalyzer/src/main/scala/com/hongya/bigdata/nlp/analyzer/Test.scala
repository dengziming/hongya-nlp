package com.hongya.bigdata.nlp.analyzer

import org.apache.spark.ml.feature.{CountVectorizer, IDF}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
  * Created by dengziming on 14/07/2017.
  * ${Main}
  */
object Test {

  def main(args: Array[String]): Unit = {

    val builder = SparkSession
      .builder
      .appName("nlp")
      .master("local[2]")
    val spark = builder.getOrCreate()

    test3(spark)

  }

  def test1(spark: SparkSession): Unit = {

    val util = new AssembleDocumentTermMatrix(spark)
    val lines = util.parseWikipediaDump("/Users/dengziming/Documents/hongya/data/day10/test.xml")
    lines.take(1).foreach(println)
  }

  def test2(spark: SparkSession): Unit ={

    val util = new AssembleDocumentTermMatrix(spark)
    val lines = util.parseWikipediaDump("/Users/dengziming/Documents/hongya/data/day10/test.xml")
    val terms = util.contentsToTerms(lines,"wikianalyzer/src/main/resources/stopwords.txt")

    terms.take(10).foreach(println)
  }


  def test3(spark: SparkSession): Unit ={
    import spark.implicits._
    val util = new AssembleDocumentTermMatrix(spark)
    val lines = util.parseWikipediaDump("/Users/dengziming/Documents/hongya/data/day10/test.xml")

    val terms = util.contentsToTerms(lines,"wikianalyzer/src/main/resources/stopwords.txt")
    val termsDF = terms.toDF("title", "terms")
    println(termsDF.count())
    val filtered = termsDF.where(size($"terms") > 1)


    // 将文档切分成词法:
    val countVectorizer = new CountVectorizer()
      .setInputCol("terms").setOutputCol("termFreqs").setVocabSize(20)
    val vocabModel = countVectorizer.fit(filtered)
    val docTermFreqs = vocabModel.transform(filtered)

    println("下面是文档的词项及频率")
    docTermFreqs.show()

    val termIds = vocabModel.vocabulary
    println("下面是所有的词项")
    termIds.foreach(println)

    docTermFreqs.cache()

    // 并给每个词项一个单独的id：
    val docIds = docTermFreqs.rdd.map(_.getString(0)).zipWithUniqueId().map(_.swap).collect().toMap
    println("所有的文档对应的id")
    docIds.foreach(println)

    // 然后利用spark自带的IDF模型，计算文档的TF-IDF
    val idf = new IDF().setInputCol("termFreqs").setOutputCol("tfidfVec")
    val idfModel = idf.fit(docTermFreqs)
    val docTermMatrix = idfModel.transform(docTermFreqs).select("title", "tfidfVec")
    // 矩阵
    docTermMatrix.show()
  }

}
