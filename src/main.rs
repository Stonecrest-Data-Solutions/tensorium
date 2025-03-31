use tensorium::{ Tensor };
use tensorium::tensor_ops:: {
    get_dimension,
    zero_tensor,
    add_tensors,
    subtract_tensors,
    divide_tensors,
    multiply_tensors,
    remainder_tensors
};

fn main() {
    let x = Tensor::Array(Vec::from([
        Tensor::Array(Vec::from([
            Tensor::Element(Vec::from([1.0, 2.0, 3.0, 1.0])),
            Tensor::Element(Vec::from([4.0, 5.0, 6.0, 1.0])),
            Tensor::Element(Vec::from([7.0, 8.0, 9.0, 1.0]))
        ])),
        Tensor::Array(Vec::from([
            Tensor::Element(Vec::from([10.0, 11.0, 12.0, 1.0])),
            Tensor::Element(Vec::from([13.0, 14.0, 15.0, 1.0])),
            Tensor::Element(Vec::from([16.0, 17.0, 18.0, 1.0]))
        ]))
    ])

    );


    println!("{x:#?}");
    let dimension = get_dimension(&x);
    println!("Dimensions: {dimension:?}");

    let z_tensor = zero_tensor(dimension);
    println!("{z_tensor:#?}");

    let a = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([1.0, 2.0])),
        Tensor::Element(Vec::from([3.0, 4.0])),
    ]));
    let b = Tensor::Array(Vec::from([
        Tensor::Element(Vec::from([4.0, 3.0])),
        Tensor::Element(Vec::from([2.0, 1.0])),
    ]));

    let sum = add_tensors(&a, &b);
    println!("Sum: {sum:?}");
    let dif = subtract_tensors(&a, &b);
    println!("Difference: {dif:?}");
    let prod = multiply_tensors(&a, &b);
    println!("Product: {prod:?}");
    let quot = divide_tensors(&a, &b);
    println!("Quotient: {quot:?}");
    let rem = remainder_tensors(&a, &b);
    println!("Remainder: {rem:?}");

    println!("Test Add: {:?}", a + b);
}
