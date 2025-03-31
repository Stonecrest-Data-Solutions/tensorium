use crate::Tensor;

/// Runs a piecewise function on between each element of two Tensors. The Tensors **must** be the
/// same shape and of the same dimensionality. If they are not, see broadcasting under
/// [crate::tensor_ops]. This function forms the foundation for all element-wise math in the
/// tensorium library.
///
/// # Examples
///
/// ```
/// use tensorium::Tensor;
/// use tensorium::tensor_ops::tensor_op;
///
/// let t1 = Tensor::Element(vec![1.0, 2.0]);
/// let t2 = Tensor::Element(vec![3.0, 4.0]);
/// let t3 = tensor_op(&t1, &t2, |x, y| x + y);
///
/// assert_eq!(
///     t3,
///     Tensor::Element(vec![4.0, 6.0])
/// );
/// ```
///
/// # Panics
///
/// This function will panic if the dimensions between the two Tensors do not match.
///
/// ```should_panic
/// use tensorium::Tensor;
/// use tensorium::tensor_ops::tensor_op;
///
/// // t1 is of shape [2], t2 is of shape [1, 3]
/// let t1 = Tensor::Element(vec![1.0, 2.0]);
/// let t2 = Tensor::Array(Vec::from([Tensor::Element(vec![1.0, 2.0, 3.0])]));
/// let t3 = tensor_op(&t1, &t2, |x, y| x + y);
/// ```
///
/// If you need to do element-wise operations on Tensor that do not match, see the section on
/// broadcasting.
pub fn tensor_op(ltensor: &Tensor, rtensor: &Tensor, func: fn(f64, f64) -> f64) -> Tensor {
    match ltensor {
        Tensor::Element(x) =>{
            match rtensor {
                Tensor::Element(y) => {
                    if x.len() == y.len() {
                        let mut outtensor: Vec<f64> = Vec::new();
                        for n in 0..x.len() {
                            outtensor.push(func(x[n], y[n]));
                        }
                        Tensor::Element(outtensor)
                    } else {
                        panic!("Element Tensors are different lengths!");
                    }
                },
                Tensor::Array(_) => panic!("Dimensionality Mismatch!")
            }
        },
        Tensor::Array(x) => {
            match rtensor {
                Tensor::Element(_) => panic!("Dimensionality Mismatch!"),
                Tensor::Array(y) => {
                    if x.len() == y.len() {
                        let mut outtensor: Vec<Tensor> = Vec::new();
                        for n in 0..x.len() {
                            outtensor.push(tensor_op(&x[n], &y[n], func));
                        }
                        Tensor::Array(outtensor)
                    } else {
                        panic!("Element Tensors are different lengths!");
                    }
                }
            }
        }
    }
}

/// Adds two Tensors element-wise. This uses [crate::tensor_ops::tensor_op()] under the hood so
/// it can panic in the same way as that function.
///
/// # Examples
///
/// ```
/// use tensorium::Tensor;
/// use tensorium::tensor_ops::add_tensors;
///
/// let t1 = Tensor::Element(vec![1.0, 2.0]);
/// let t2 = Tensor::Element(vec![3.0, 4.0]);
/// let t3 = add_tensors(&t1, &t2);
///
/// assert_eq!(t3, Tensor::Element(vec![4.0, 6.0]));
/// ```
pub fn add_tensors(ltensor: &Tensor, rtensor: &Tensor) -> Tensor {
    tensor_op(ltensor, rtensor, |x, y| x + y)
}

/// Subtracts two Tensors element-wise. This uses [crate::tensor_ops::tensor_op()] under the hood
/// so it can panic in the same way as that function.
///
/// # Examples
///
/// ```
/// use tensorium::Tensor;
/// use tensorium::tensor_ops::subtract_tensors;
///
/// let t1 = Tensor::Element(vec![1.0, 2.0]);
/// let t2 = Tensor::Element(vec![3.0, 4.0]);
/// let t3 = subtract_tensors(&t1, &t2);
///
/// assert_eq!(t3, Tensor::Element(vec![-2.0, -2.0]));
/// ```
pub fn subtract_tensors(ltensor: &Tensor, rtensor: &Tensor) -> Tensor {
    tensor_op(ltensor, rtensor, |x, y| x - y)
}

/// Multiplies two Tensors element-wise. This uses [crate::tensor_ops::tensor_op()] under the
/// hood so it can panic in the same way as that function.
///
/// # Examples
///
/// ```
/// use tensorium::Tensor;
/// use tensorium::tensor_ops::multiply_tensors;
///
/// let t1 = Tensor::Element(vec![1.0, 2.0]);
/// let t2 = Tensor::Element(vec![3.0, 4.0]);
/// let t3 = multiply_tensors(&t1, &t2);
///
/// assert_eq!(t3, Tensor::Element(vec![3.0, 8.0]));
/// ```
pub fn multiply_tensors(ltensor: &Tensor, rtensor: &Tensor) -> Tensor {
    tensor_op(ltensor, rtensor, |x, y| x * y)
}

/// Divides two Tensors element-wise. This uses [crate::tensor_ops::tensor_op()] under the hood
/// so it can panic in the same way as that function.
///
/// # Examples
///
/// ```
/// use tensorium::Tensor;
/// use tensorium::tensor_ops::divide_tensors;
///
/// let t1 = Tensor::Element(vec![1.0, 2.0]);
/// let t2 = Tensor::Element(vec![2.0, 4.0]);
/// let t3 = divide_tensors(&t1, &t2);
///
/// assert_eq!(t3, Tensor::Element(vec![0.5, 0.5]));
/// ```
pub fn divide_tensors(ltensor: &Tensor, rtensor: &Tensor) -> Tensor {
    tensor_op(ltensor, rtensor, |x, y| x / y)
}

/// Remainder of two Tensors element-wise. This uses [crate::tensor_ops::tensor_op] under the
/// hood so it can panic in the same way as that function.
///
/// # Examples
///
/// ```
/// use tensorium::Tensor;
/// use tensorium::tensor_ops::remainder_tensors;
///
/// let t1 = Tensor::Element(vec![7.0, 5.0]);
/// let t2 = Tensor::Element(vec![3.0, 4.0]);
/// let t3 = remainder_tensors(&t1, &t2);
///
/// assert_eq!(t3, Tensor::Element(vec![1.0, 1.0]));
/// ```
pub fn remainder_tensors(ltensor: &Tensor, rtensor: &Tensor) -> Tensor {
    tensor_op(ltensor, rtensor, |x, y| x % y)
}