package com.hongya.bigdata.nlp.analyzer

import java.io.StringReader

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.wltea.analyzer.core.{IKSegmenter, Lexeme}

import scala.collection.mutable.ArrayBuffer


object Comment {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("comment").setMaster("local[2]")
    val sc = new SparkContext(conf)

    val originData = sc.textFile("analyzer/src/main/resources/comment.txt")
    val originDistinctData = originData.distinct()
    val rateDocument = originDistinctData.map( line => line.split("\\s+")).filter(line => line.length == 2)

    val fiveRateDocument = rateDocument.filter(line => line(0) == "5")

    val oneRateDocument = rateDocument.filter(line => line(0) == "1")
    val twoRateDocument = rateDocument.filter(line => line(0) == "2")
    val threeRateDocument = rateDocument.filter(line => line(0) == "3")
    val negRateDocument = oneRateDocument.union(twoRateDocument).union(threeRateDocument)
    negRateDocument.repartition(1)
    negRateDocument.take(10).foreach(println)

    val posRateDocument = fiveRateDocument
    posRateDocument.take(10).foreach(println)

    val allRateDocument = negRateDocument.union(posRateDocument)
    allRateDocument.repartition(1)

    // 打分和评论
    val rate = allRateDocument.map( s => s(0).toDouble)
    val document = allRateDocument.map(s => s(1))

    //分词
    val words = document.map(cut)

    val hashingTF = new HashingTF()
    val tf = hashingTF.transform(words)
    tf.cache()
    val idf = new IDF().fit(tf)
    val tfidf: RDD[Vector] = idf.transform(tf)

    // spark.mllib IDF implementation provides an option for ignoring terms which occur in less than
    // a minimum number of documents. In such cases, the IDF for these terms is set to 0.
    // This feature can be used by passing the minDocFreq value to the IDF constructor.

    val zipped = rate.zip(tfidf)
    val data = zipped.map( line => LabeledPoint(line._1,line._2))
    val Array(training,test) = data.randomSplit(Array(0.7, 0.3), seed = 0)

    val NBmodel = NaiveBayes.train(training, 1.0)
    val predictionAndLabel = test.map(p => (NBmodel.predict(p.features), p.label))
    val accuracy = predictionAndLabel.filter( x => if (x._1 == x._2) true else false ).count() / test.count()

    println(NBmodel.predict(hashingTF.transform(cut("这是垃圾烂片啊"))) )
    println(NBmodel.predict(hashingTF.transform(cut("很好的电影啊"))) )

  }

  def cut(text : String): Seq[String] = {

    val result = ArrayBuffer[String]()
    val ik = new IKSegmenter(new StringReader(text), true)

    var word:Lexeme = ik.next()
    while(word != null) {
      result += word.getLexemeText
      word = ik.next()
    }
    result
  }

}