use crate::Tensor;
use crate::tensor_ops;
use crate::tensor_ops::{broadcast_tensor, expand_dims, get_dimension};

#[test]
fn basic_broadcasting() {
    let a = Tensor::Array(Vec::from([Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([1.0, 2.0, 3.0])),
        Tensor::Element(Vec::from([4.0, 5.0, 6.0])),
        Tensor::Element(Vec::from([7.0, 8.0, 9.0]))
    ]))]));

    let target_shape: Vec<usize> = Vec::from([2, 3, 3]);

    let broadcasted_tensor = tensor_ops::broadcast_tensor(&a, &target_shape);
    let truth_tensor = Tensor::Array(Vec::from([
        Tensor::Array(Vec::from([
            Tensor::Element(Vec::from([1.0, 2.0, 3.0])),
            Tensor::Element(Vec::from([4.0, 5.0, 6.0])),
            Tensor::Element(Vec::from([7.0, 8.0, 9.0]))
        ])),
        Tensor::Array(Vec::from([
            Tensor::Element(Vec::from([1.0, 2.0, 3.0])),
            Tensor::Element(Vec::from([4.0, 5.0, 6.0])),
            Tensor::Element(Vec::from([7.0, 8.0, 9.0]))
        ]))
    ]));

    assert_eq!(broadcasted_tensor, truth_tensor);
}


#[test]
fn expanding_dims() {
    let a = Tensor::Element(Vec::from([1.0, 2.0, 3.0]));
    let target_dims: Vec<usize> = Vec::from([1, 1, 3]);

    let expanded_a = get_dimension(&expand_dims(a, 2));

    assert_eq!(expanded_a, target_dims);
}

#[test]
fn singular_broadcasting() {
    let a = Tensor::Element(Vec::from([1.0]));
    let target_dims = Vec::from([3, 3, 3]);

    let num_expansions = target_dims.len() - get_dimension(&a).len();
    let aa = tensor_ops::expand_dims(a, num_expansions);

    let target_broadcast = Tensor::Array(Vec::from([
        Tensor::Array(Vec::from([
            Tensor::Element(Vec::from([1.0, 1.0, 1.0])),
            Tensor::Element(Vec::from([1.0, 1.0, 1.0])),
            Tensor::Element(Vec::from([1.0, 1.0, 1.0])),
        ])),
        Tensor::Array(Vec::from([
            Tensor::Element(Vec::from([1.0, 1.0, 1.0])),
            Tensor::Element(Vec::from([1.0, 1.0, 1.0])),
            Tensor::Element(Vec::from([1.0, 1.0, 1.0])),
        ])),
        Tensor::Array(Vec::from([
            Tensor::Element(Vec::from([1.0, 1.0, 1.0])),
            Tensor::Element(Vec::from([1.0, 1.0, 1.0])),
            Tensor::Element(Vec::from([1.0, 1.0, 1.0])),
        ])),
    ]));

    let broadcasted_aa = broadcast_tensor(&aa, &target_dims);

    assert_eq!(broadcasted_aa, target_broadcast);
}