use crate::tensor_ops;
use crate::tensor_objects::Tensor;

#[test]
fn test_sum() {
    let t1 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([1.0, 1.0])),
        Tensor::Element(Vec::from([2.0, 3.0]))
    ]));
    let t2 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([1.0, 1.0])),
        Tensor::Element(Vec::from([2.0, 3.0]))
    ]));
    let t3 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([2.0, 2.0])),
        Tensor::Element(Vec::from([4.0, 6.0]))
    ]));

    assert_eq!(
        tensor_ops::add_tensors(&t1, &t2),
        t3
    )
}

#[test]
fn test_sub() {
    let t1 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([1.0, 1.0])),
        Tensor::Element(Vec::from([2.0, 3.0]))
    ]));
    let t2 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([1.0, 1.0])),
        Tensor::Element(Vec::from([2.0, 3.0]))
    ]));
    let t3 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([0.0, 0.0])),
        Tensor::Element(Vec::from([0.0, 0.0]))
    ]));

    assert_eq!(
        tensor_ops::subtract_tensors(&t1, &t2),
        t3
    )
}

#[test]
fn test_mul() {
    let t1 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([1.0, 1.0])),
        Tensor::Element(Vec::from([2.0, 3.0]))
    ]));
    let t2 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([1.0, 1.0])),
        Tensor::Element(Vec::from([2.0, 3.0]))
    ]));
    let t3 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([1.0, 1.0])),
        Tensor::Element(Vec::from([4.0, 9.0]))
    ]));

    assert_eq!(
        tensor_ops::multiply_tensors(&t1, &t2),
        t3
    )
}

#[test]
fn test_div() {
    let t1 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([2.0, 4.0])),
        Tensor::Element(Vec::from([9.0, 20.0]))
    ]));
    let t2 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([1.0, 2.0])),
        Tensor::Element(Vec::from([3.0, 5.0]))
    ]));
    let t3 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([2.0, 2.0])),
        Tensor::Element(Vec::from([3.0, 4.0]))
    ]));

    assert_eq!(
        tensor_ops::divide_tensors(&t1, &t2),
        t3
    )
}

#[test]
fn test_rem() {
    let t1 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([5.0, 5.0])),
        Tensor::Element(Vec::from([3.0, 3.0]))
    ]));
    let t2 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([4.0, 2.0])),
        Tensor::Element(Vec::from([2.0, 1.5]))
    ]));
    let t3 = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([1.0, 1.0])),
        Tensor::Element(Vec::from([1.0, 0.0]))
    ]));

    assert_eq!(
        tensor_ops::remainder_tensors(&t1, &t2),
        t3
    )
}

#[test]
fn same_dimensions() {
    let dim1: Vec<usize> = Vec::from([2, 2]);
    let dim2: Vec<usize> = Vec::from([2, 2]);

    assert!(tensor_ops::is_broadcastable(&dim1, &dim2));
}

#[test]
fn one_first_dimension() {
    let dim1: Vec<usize> = Vec::from([2, 1]);
    let dim2: Vec<usize> = Vec::from([2, 2]);

    assert!(tensor_ops::is_broadcastable(&dim1, &dim2));
}

#[test]
fn different_lengths() {
    let dim1: Vec<usize> = Vec::from([3, 5, 7, 9]);
    let dim2: Vec<usize> = Vec::from([1, 9]);

    assert!(tensor_ops::is_broadcastable(&dim1, &dim2));
}

#[test]
fn not_broadcastable() {
    let dim1: Vec<usize> = Vec::from([3, 5]);
    let dim2: Vec<usize> = Vec::from([3, 2, 1]);

    assert!(!tensor_ops::is_broadcastable(&dim1, &dim2));
}

#[test]
fn broadcast_test() {
    let dim1: Vec<usize> = Vec::from([4, 1, 5]);
    let dim2: Vec<usize> = Vec::from([4, 5]);

    let dim12 = tensor_ops::broadcast_shape(&dim1, &dim2);

    assert_eq!(dim12, Vec::from([4, 4, 5]));
}

#[test]
fn expand_test() {
    let tensor: Tensor = Tensor::Element(Vec::from([1.0, 2.0, 3.0]));
    let truth_tensor: Tensor = Tensor::Array(Vec::from(
        [
            Tensor::Element(Vec::from([1.0, 2.0, 3.0])),
            Tensor::Element(Vec::from([1.0, 2.0, 3.0])),
            Tensor::Element(Vec::from([1.0, 2.0, 3.0]))
        ]
    ));
    let copy_num: u32 = 3;

    assert_eq!(
        tensor_ops::expand_tensor(&tensor, copy_num),
        truth_tensor
    );
}