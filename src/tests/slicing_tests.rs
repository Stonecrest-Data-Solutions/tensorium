use crate::Tensor;
use crate::tensor_objects::TensorIndexResult;

#[test]
fn test_single_index_array() {
    let a = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([1.0, 2.0, 3.0])),
        Tensor::Element(Vec::from([4.0, 5.0, 6.0])),
        Tensor::Element(Vec::from([7.0, 8.0, 9.0])),
    ]));

    let a_slice = a.index(0).unwrap();

    match a_slice {
        TensorIndexResult::Tensor(t) => {
            assert_eq!(
                t,
                Tensor::Element(Vec::from([1.0, 2.0, 3.0]))
            )
        },
        _ => panic!("Did not receive a TensorIndexResult::Tensor")
    }

}

#[test]
fn test_single_index_element() {
    let a = Tensor::Element(Vec::from([1.0, 2.0, 3.0]));

    let a_slice = a.index(0).unwrap();

    match a_slice {
        TensorIndexResult::Value(t) => assert_eq!(t, 1.0),
        _ => panic!("Did not receive a TensorIndexResult::Value")
    }
}

#[test]
fn test_array_slicing() {
    let a = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([1.0, 2.0, 3.0])),
        Tensor::Element(Vec::from([4.0, 5.0, 6.0])),
        Tensor::Element(Vec::from([7.0, 8.0, 9.0])),
    ]));

    let a_slice = a.slice(0..2);

    assert_eq!(
        a_slice,
        Tensor::Array(Vec::from([
            Tensor::Element(Vec::from([1.0, 2.0, 3.0])),
            Tensor::Element(Vec::from([4.0, 5.0, 6.0]))
        ]))
    )
}

#[test]
fn test_element_slicing() {
    let a = Tensor::Element(Vec::from([1.0, 2.0, 3.0]));

    let a_slice = a.slice(0..2);

    assert_eq!(
        a_slice,
        Tensor::Element(Vec::from([1.0, 2.0]))
    )
}