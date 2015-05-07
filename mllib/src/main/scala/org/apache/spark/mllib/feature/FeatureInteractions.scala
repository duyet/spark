package org.apache.spark.mllib.feature

import org.apache.spark.mllib.linalg.{Vectors, Vector}

/**
 *
 * Adds 2-nd order interactions to a given vector.
 *
 * Given a vector of features [x_i] it transforms it into that [x_i] ++ [x_i * x_j] for all i <= j
 * Order is: original features then x_i * x_j in increasing pair order of indices (i, j)
 *
 * E.g. [2.0, 3.0, 7.0] => [2.0, 3.0, 7.0, 4.0, 9.0, 49.0, 6.0, 14.0, 21.0]
 *
 */
class FeatureInteractions extends VectorTransformer {
  /**
   * Applies transformation on a vector.
   *
   * @param vector vector to be transformed.
   * @return transformed vector.
   */
  override def transform(vector: Vector): Vector = {
    assert(vector.size > 0, "feature vector must have at least one feature to add interactions")

    val v = vector.toBreeze
    val interactions = (for {
      (i, a) <- v.iterator
      (j, b) <- v.iterator if i <= j
    } yield a * b).toArray

    Vectors.dense(v.toArray ++ interactions)
  }
}
