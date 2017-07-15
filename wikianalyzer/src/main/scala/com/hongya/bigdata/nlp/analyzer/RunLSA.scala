/*
 * Copyright 2015 and onwards Sanford Ryza, Uri Laserson, Sean Owen and Joshua Wills
 *
 * See LICENSE file for further information.
 */

package com.hongya.bigdata.nlp.analyzer

import breeze.linalg.{DenseMatrix => BDenseMatrix, SparseVector => BSparseVector}

import org.apache.spark.mllib.linalg.{Matrices, Matrix, SingularValueDecomposition, Vectors, Vector => MLLibVector}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer

object RunLSA {
  def main(args: Array[String]): Unit = {

    val k = if (args.length > 0) args(0).toInt else 100
    val numTerms = if (args.length > 1) args(1).toInt else 20000

    val spark = SparkSession.builder().config("spark.serializer", classOf[KryoSerializer].getName).getOrCreate()
    val assembleMatrix = new AssembleDocumentTermMatrix(spark)
    import assembleMatrix._

    val docTexts: Dataset[(String, String)] = parseWikipediaDump("")

    val (docTermMatrix, termIds, docIds, termIdfs) = documentTermMatrix(docTexts, "stopwords.txt", numTerms)

    docTermMatrix.cache()

    val vecRdd = docTermMatrix.select("tfidfVec").rdd.map { row =>
      Vectors.fromML(row.getAs[MLVector]("tfidfVec"))
    }

    vecRdd.cache()
    val mat = new RowMatrix(vecRdd)
    val svd = mat.computeSVD(k, computeU=true)

    println("Singular values: " + svd.s)
    val topConceptTerms = topTermsInTopConcepts(svd, 10, 10, termIds)
    val topConceptDocs = topDocsInTopConcepts(svd, 10, 10, docIds)
    for ((terms, docs) <- topConceptTerms.zip(topConceptDocs)) {
      println("Concept terms: " + terms.map(_._1).mkString(", "))
      println("Concept docs: " + docs.map(_._1).mkString(", "))
      println()
    }

    val queryEngine = new LSAQueryEngine(svd, termIds, docIds, termIdfs)
    queryEngine.printTopTermsForTerm("algorithm")
    queryEngine.printTopTermsForTerm("radiohead")
    queryEngine.printTopTermsForTerm("tarantino")

    queryEngine.printTopDocsForTerm("fir")
    queryEngine.printTopDocsForTerm("graph")

    queryEngine.printTopDocsForDoc("Romania")
    queryEngine.printTopDocsForDoc("Brad Pitt")
    queryEngine.printTopDocsForDoc("Radiohead")

    queryEngine.printTopDocsForTermQuery(Seq("factorization", "decomposition"))
  }

  /**
   * 最重要的概念相关的词项，每个概念，得到最相关的词项。
   *
   * @param svd  svd矩阵
   * @param numConcepts 希望查看的concept数
   * @param numTerms 希望查看的词项数.
   * @param termIds  IDs 到 terms 的隐射
   * @return 最重要的概念
   */
  def topTermsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int,
      numTerms: Int, termIds: Array[String]): Seq[Seq[(String, Double)]] = {
    val v = svd.V
    val topTerms = new ArrayBuffer[Seq[(String, Double)]]()
    val arr = v.toArray
    for (i <- 0 until numConcepts) {
      val offs = i * v.numRows
      val termWeights = arr.slice(offs, offs + v.numRows).zipWithIndex
      val sorted = termWeights.sortBy(-_._1)
      topTerms += sorted.take(numTerms).map {case (score, id) => (termIds(id), score) }
    }
    topTerms
  }

  /**
   * 得到和重要概念相关的词项，但是因为它是分布式的我们的逻辑稍有不同
   *
   *  @param svd  svd矩阵
    * @param numConcepts 希望查看的concept数
    * @param numDocs 希望查看的文档数.
    * @param docIds  IDs 到 docs 的隐射
    * @return 最重要的概念
   */
  def topDocsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int,
      numDocs: Int, docIds: Map[Long, String]): Seq[Seq[(String, Double)]] = {
    val u  = svd.U
    val topDocs = new ArrayBuffer[Seq[(String, Double)]]()
    for (i <- 0 until numConcepts) {
      val docWeights = u.rows.map(_.toArray(i)).zipWithUniqueId
      topDocs += docWeights.top(numDocs).map { case (score, id) => (docIds(id), score) }
    }
    topDocs
  }
}

class LSAQueryEngine(
    val svd: SingularValueDecomposition[RowMatrix, Matrix],
    val termIds: Array[String],
    val docIds: Map[Long, String],
    val termIdfs: Array[Double]) {

  val VS: BDenseMatrix[Double] = multiplyByDiagonalMatrix(svd.V, svd.s)
  val normalizedVS: BDenseMatrix[Double] = rowsNormalized(VS)
  val US: RowMatrix = multiplyByDiagonalRowMatrix(svd.U, svd.s)
  val normalizedUS: RowMatrix = distributedRowsNormalized(US)

  val idTerms: Map[String, Int] = termIds.zipWithIndex.toMap
  val idDocs: Map[String, Long] = docIds.map(_.swap)

  /**
   * 得到矩阵和对角矩阵的乘积，由于没有实现，我们需要自己实现
   */
  def multiplyByDiagonalMatrix(mat: Matrix, diag: MLLibVector): BDenseMatrix[Double] = {
    val sArr = diag.toArray
    new BDenseMatrix[Double](mat.numRows, mat.numCols, mat.toArray)
      .mapPairs { case ((r, c), v) => v * sArr(c) }
  }

  /**
   * 得到分布式矩阵和对角矩阵的乘积，由于没有实现，我们需要自己实现
   */
  def multiplyByDiagonalRowMatrix(mat: RowMatrix, diag: MLLibVector): RowMatrix = {
    val sArr = diag.toArray
    new RowMatrix(mat.rows.map { vec =>
      val vecArr = vec.toArray
      val newArr = (0 until vec.size).toArray.map(i => vecArr(i) * sArr(i))
      Vectors.dense(newArr)
    })
  }

  /**
   * 矩阵每一行都除以它的长度，归一化
   */
  def rowsNormalized(mat: BDenseMatrix[Double]): BDenseMatrix[Double] = {
    val newMat = new BDenseMatrix[Double](mat.rows, mat.cols)
    for (r <- 0 until mat.rows) {
      val length = math.sqrt((0 until mat.cols).map(c => mat(r, c) * mat(r, c)).sum)
      (0 until mat.cols).foreach(c => newMat.update(r, c, mat(r, c) / length))
    }
    newMat
  }

  /**
   * 分布式矩阵每一行都除以它的长度，归一化
   */
  def distributedRowsNormalized(mat: RowMatrix): RowMatrix = {
    new RowMatrix(mat.rows.map { vec =>
      val array = vec.toArray
      val length = math.sqrt(array.map(x => x * x).sum)
      Vectors.dense(array.map(_ / length))
    })
  }

  /**
   * 每个词项相关的文档，返回id和相关度
   */
  def topDocsForTerm(termId: Int): Seq[(Double, Long)] = {
    val rowArr = (0 until svd.V.numCols).map(i => svd.V(termId, i)).toArray
    val rowVec = Matrices.dense(rowArr.length, 1, rowArr)

    // 每个文档的相关度
    val docScores = US.multiply(rowVec)

    // 最高分的文档
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
    allDocWeights.top(10)
  }

  /**
   * 和词项相关度最高的词项
   */
  def topTermsForTerm(termId: Int): Seq[(Double, Int)] = {
    // 查看VS每一行和termId最相关的
    val rowVec = normalizedVS(termId, ::).t

    // 和每个term相关的
    val termScores = (normalizedVS * rowVec).toArray.zipWithIndex

    // 找到得分最高的
    termScores.sortBy(-_._1).take(10)
  }

  /**
   * 文档相关度最高的的文档
   */
  def topDocsForDoc(docId: Long): Seq[(Double, Long)] = {
    //
    val docRowArr = normalizedUS.rows.zipWithUniqueId.map(_.swap).lookup(docId).head.toArray
    val docRowVec = Matrices.dense(docRowArr.length, 1, docRowArr)

    //
    val docScores = normalizedUS.multiply(docRowVec)

    //
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId

    //
    allDocWeights.filter(!_._1.isNaN).top(10)
  }

  /**
    * 将一系列的词项转化为向量
    */
  def termsToQueryVector(terms: Seq[String]): BSparseVector[Double] = {
    val indices = terms.map(idTerms(_)).toArray
    val values = indices.map(termIdfs(_))
    new BSparseVector[Double](indices, values, idTerms.size)
  }

  /**
    * 词项和文档相关度
    */
  def topDocsForTermQuery(query: BSparseVector[Double]): Seq[(Double, Long)] = {
    val breezeV = new BDenseMatrix[Double](svd.V.numRows, svd.V.numCols, svd.V.toArray)
    val termRowArr = (breezeV.t * query).toArray

    val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

    //
    val docScores = US.multiply(termRowVec)

    //
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
    allDocWeights.top(10)
  }

  def printTopTermsForTerm(term: String): Unit = {
    val idWeights = topTermsForTerm(idTerms(term))
    println(idWeights.map { case (score, id) => (termIds(id), score) }.mkString(", "))
  }

  def printTopDocsForDoc(doc: String): Unit = {
    val idWeights = topDocsForDoc(idDocs(doc))
    println(idWeights.map { case (score, id) => (docIds(id), score) }.mkString(", "))
  }

  def printTopDocsForTerm(term: String): Unit = {
    val idWeights = topDocsForTerm(idTerms(term))
    println(idWeights.map { case (score, id) => (docIds(id), score) }.mkString(", "))
  }

  def printTopDocsForTermQuery(terms: Seq[String]): Unit = {
    val queryVec = termsToQueryVector(terms)
    val idWeights = topDocsForTermQuery(queryVec)
    println(idWeights.map { case (score, id) => (docIds(id), score) }.mkString(", "))
  }
}
