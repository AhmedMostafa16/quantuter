use ndarray::{linalg::kron, Array1, Array2};
use num_complex::{Complex, Complex64};
use rand::{distributions::Distribution, SeedableRng};
use rand_hc::Hc128Rng;
use std::str::Chars;
use std::{
    collections::BTreeMap,
    env,
    f64::consts::E,
    fs::File,
    io::{self, Read},
};

/// A quantum statevector holding `size` qubits
#[derive(Clone, Debug)]
pub struct QunatumCurcuit {
    size: usize,
    state: Array1<Complex64>,
    identities: Vec<Array2<Complex64>>,
}

impl QunatumCurcuit {
    /// Creates a new QunatumCurcuit instance with `size` qubits
    pub fn new(size: usize) -> Self {
        let mut state: Array1<Complex64> = Array1::zeros(1 << size);
        state[0] = Complex64::new(1.0, 0.0);

        // Create identities so I don't have to re-compute every time
        let identities = (0..size).map(|i| Array2::eye(1 << i)).collect::<Vec<_>>();

        Self {
            size,
            state,
            identities,
        }
    }

    /// Compute a single gate operator with the given 2x2 matrix and 0-based index
    pub fn apply_gate(&mut self, operator: Array2<Complex64>, x: usize) {
        assert!(operator.dim() == (2, 2));

        // Create the state vector for the operator
        let size = self.size - x - 1;
        let mut state = self.identities[size].clone();
        state = kron(&state, &operator);

        // Create the state vector for the rest of the qubits
        let state2 = &self.identities[x];
        state = kron(&state, state2);

        // Multiply state with self.state, while also storing in self.state
        self.state = state.dot(&self.state);
    }

    /// Identity gate
    pub fn identity(&mut self, x: usize) {
        let operator = Array2::eye(2);
        self.apply_gate(operator, x);
    }

    /// Hadamard gate
    pub fn hadamard(&mut self, x0: usize) {
        let sqrt2inv = 1.0 / 2.0f64.sqrt();
        let operator = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(sqrt2inv, 0.0),
                Complex::new(sqrt2inv, 0.0),
                Complex::new(sqrt2inv, 0.0),
                Complex::new(-sqrt2inv, 0.0),
            ],
        )
        .expect("Failed to create Hadamard gate");

        self.apply_gate(operator, x0);
    }

    /// X / NOT gate
    pub fn x(&mut self, x: usize) {
        let operator = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(0.0, -1.0),
                Complex::new(0.0, 1.0),
                Complex::new(0.0, 0.0),
            ],
        )
        .expect("Failed to create X gate");
        self.apply_gate(operator, x)
    }

    /// Y gate
    pub fn y(&mut self, x: usize) {
        let operator = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(0.0, -1.0),
                Complex::new(0.0, 1.0),
                Complex::new(0.0, 0.0),
            ],
        )
        .expect("Failed to create Y gate");

        self.apply_gate(operator, x);
    }

    /// Z gate
    pub fn z(&mut self, x: usize) {
        let operator = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(-1.0, 0.0),
            ],
        )
        .expect("Faild to create Z gate");

        self.apply_gate(operator, x);
    }

    /// CX / CNOT gate with a given 0-based control and target qubit
    pub fn cnot(&mut self, control: usize, target: usize) {
        let braket0 = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
        )
        .unwrap();
        let braket1 = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
            ],
        )
        .unwrap();
        let x = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
        )
        .unwrap();

        let x1 = target;

        // If control qubit is greater than target run first option, otherwise reverse it
        match control {
            x0 if x0 > x1 => {
                let mut state_0 = kron(&self.identities[x1], &braket0);
                state_0 = kron(&state_0, &self.identities[self.size - x1 - 1]);

                let mut state_1 = kron(&self.identities[x1], &braket1);
                state_1 = kron(&state_1, &self.identities[x0 - x1 - 1]);
                state_1 = kron(&state_1, &x);
                state_1 = kron(&state_1, &self.identities[self.size - x0 - 1]);

                let state = state_0 + state_1;
                // general_mat_vec_mul(1.0, &state, &self.state.clone(), 0.0, &mut self.state);
                self.state = state.dot(&self.state);
            }
            x0 if x0 < x1 => {
                let mut state_0 = kron(&braket0, &self.identities[x0]);
                state_0 = kron(&self.identities[self.size - x0 - 1], &state_0);

                let mut state_1 = kron(&braket1, &self.identities[x0]);
                state_1 = kron(&self.identities[x1 - x0 - 1], &state_1);
                state_1 = kron(&x, &state_1);
                state_1 = kron(&self.identities[self.size - x1 - 1], &state_1);

                let state = state_0 + state_1;
                // general_mat_vec_mul(1.0, &state, &self.state.clone(), 0.0, &mut self.state);
                self.state = state.dot(&self.state);
            }
            _ => panic!("Cannot use same control and target qubit with CX gate"),
        }
    }

    pub fn rx(&mut self, x: usize, theta: f64) {
        let operator = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(theta.cos() / 2.0, 0.0),
                Complex::new(0.0, -1.0f64 * theta.sin() / 2.0),
                Complex::new(0.0, -1.0f64 * theta.sin() / 2.0),
                Complex::new(theta.cos() / 2.0, 0.0),
            ],
        )
        .expect("Failed to create Rx gate");

        self.apply_gate(operator, x);
    }

    pub fn ry(&mut self, x: usize, theta: f64) {
        let operator = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(theta.cos() / 2.0, 0.0),
                Complex::new(-theta.sin() / 2.0, 0.0),
                Complex::new(-theta.sin() / 2.0, 0.0),
                Complex::new(theta.cos() / 2.0, 0.0),
            ],
        )
        .expect("Failed to create Ry gate");

        self.apply_gate(operator, x);
    }

    pub fn rz(&mut self, x: usize, theta: f64) {
        let operator = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(0.0, E.powf(-theta.sin() / 2.0f64)),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, E.powf(theta.sin() / 2.0f64)),
            ],
        )
        .expect("Failed to create Rz gate");

        self.apply_gate(operator, x);
    }

    /// Swap gate
    pub fn swap(&mut self, x: usize, y: usize) {
        self.cnot(x, y);
        self.cnot(y, x);
    }

    /// Quantum fourier transform
    pub fn qft(&mut self) {
        for i in 0..self.size {
            self.hadamard(i);
            for j in i + 1..self.size {
                self.cnot(i, j);
                self.rx(j, std::f64::consts::PI / 2.0f64.powi((j - i) as i32));
            }
        }
    }

    fn measure(&mut self, qubit: usize) -> u32 {
        let probs = self.measure_prob();
        let rng_uni = rand::distributions::WeightedIndex::new(probs.values()).unwrap();
        let index = rng_uni.sample(&mut Hc128Rng::from_entropy());
        let bits = probs.keys().nth(index).unwrap();
        let bit = bits.chars().nth(self.size - qubit - 1).unwrap();
        bit.to_digit(10).unwrap()
    }

    /// Measures the theoretical probability of each state
    /// This is a deterministic function and will return the same results each time it is called
    pub fn measure_prob(&self) -> BTreeMap<String, f64> {
        // Generate all numbers up to 2^self.size and convert to binary
        // Since it uses u64s this means max would be 63 qubits, but that shouldnt be a problem
        // use the real part of the complex numbers
        (0..1u64 << self.size)
            .map(|i| {
                let bits = format!("{:0width$b}", i, width = self.size);
                let index = usize::from_str_radix(&bits, 2).unwrap();
                (bits, self.state[index].re.powi(2))
            })
            .collect()
    }

    /// Measures the results for a given amount of shots
    /// This is a probabilistic function and will return different results each time it is called
    /// This is a real measurement, not a theoretical one
    pub fn measure_real(&self, shots: usize) -> BTreeMap<String, usize> {
        let probs: BTreeMap<String, f64> = self.measure_prob();
        let rng_uni = rand::distributions::WeightedIndex::new(probs.values()).unwrap();
        let mut results: BTreeMap<String, usize> = probs
            .keys()
            .cloned()
            .zip(std::iter::once(0).cycle())
            .collect();

        for index in rng_uni.sample_iter(Hc128Rng::from_entropy()).take(shots) {
            *results.iter_mut().nth(index).unwrap().1 += 1
        }

        results
    }

    /// Measures a probability for a given amount of shots
    pub fn measure_real_prob(&self, shots: usize) -> BTreeMap<String, f64> {
        let real = self.measure_real(shots);
        real.into_iter()
            .map(|(bits, v)| (bits, v as f64 / shots as f64))
            .collect()
    }

    /// Prints the state of the qubits in a beautiful way
    pub fn print_state(&self) {
        let mut s: String = "".to_owned();

        let width = (self.state.len() as f64).log2() as usize;

        for (i, coef) in self.state.iter().enumerate() {
            s.push_str(&format!("|{:0width$b}⟩: {} \n", i, coef, width = width));
        }
        println!("Initial State:\n{}", s);
    }

    fn print_state_at_gate(&self, gate: &str) {
        let mut s: String = "".to_owned();

        let width = (self.state.len() as f64).log2() as usize;

        for (i, coef) in self.state.iter().enumerate() {
            s.push_str(&format!("|{:0width$b}⟩: {} \n", i, coef, width = width));
        }
        println!("State at gate '{}': \n{}", gate, s);
    }
}

#[derive(Debug, PartialEq)]
enum Token {
    Circuit(usize, Option<String>),
    Comment(String),
    Gate(String, Vec<usize>),
}

#[derive(Debug)]
#[allow(dead_code)]
enum ParseError {
    InvalidToken(String),
    UnexpectedEnd,
    MissingParameter,
}

struct QuantaParser<'a> {
    iter: Chars<'a>,
    current_char: Option<char>,
}

impl<'a> QuantaParser<'a> {
    fn new(input: &'a str) -> QuantaParser<'a> {
        let mut iter = input.chars();
        let current_char = iter.next();
        QuantaParser { iter, current_char }
    }

    fn advance(&mut self) {
        self.current_char = self.iter.next();
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.current_char {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn parse_number(&mut self) -> Result<usize, ParseError> {
        let mut num_str = String::new();

        self.skip_whitespace();

        while let Some(c) = self.current_char {
            if c.is_digit(10) {
                num_str.push(c);
                self.advance();
            } else {
                break;
            }
        }

        if num_str.is_empty() {
            Err(ParseError::MissingParameter)
        } else {
            num_str
                .parse::<usize>()
                .map_err(|_| ParseError::InvalidToken(num_str))
        }
    }

    fn parse_circuit(&mut self) -> Result<Token, ParseError> {
        for _ in 0..7 {
            self.advance(); // Skip 'c', 'i', 'r', 'c', 'u', 'i', 't'
        }

        let num_qubits = self.parse_number()?;
        let mut bitstring = None;

        self.skip_whitespace();

        // Try to parse bitstring
        if let Some(c) = self.current_char {
            if c.is_whitespace() {
                // No bitstring provided
                self.skip_whitespace();
            } else if c == '0' || c == '1' {
                let mut bitstring_chars = vec![c];
                self.advance();

                while let Some(c) = self.current_char {
                    if c.is_whitespace() {
                        break;
                    } else if c == '0' || c == '1' {
                        bitstring_chars.push(c);
                        self.advance();
                    } else {
                        return Err(ParseError::InvalidToken(bitstring_chars.iter().collect()));
                    }
                }

                bitstring = Some(bitstring_chars.iter().collect());
            }
        }

        Ok(Token::Circuit(num_qubits, bitstring))
    }

    fn parse_comment(&mut self) -> Result<Token, ParseError> {
        self.advance(); // Skip ';'

        let mut comment = String::new();
        while let Some(c) = self.current_char {
            if c == '\n' {
                break;
            }
            comment.push(c);
            self.advance();
        }

        Ok(Token::Comment(comment))
    }

    fn parse_gate(&mut self) -> Result<Token, ParseError> {
        let gate_name = self.parse_identifier()?;
        let mut qubits = Vec::new();

        self.skip_whitespace();

        while let Some(c) = self.current_char {
            if c.is_digit(10) {
                let qubit = self.parse_number()?;
                qubits.push(qubit);
            } else if c == '\n' {
                break;
            } else if c.is_whitespace() {
                self.skip_whitespace();
            } else {
                return Err(ParseError::InvalidToken(c.to_string()));
            }
        }

        Ok(Token::Gate(gate_name, qubits))
    }

    fn parse_identifier(&mut self) -> Result<String, ParseError> {
        let mut identifier = String::new();

        self.skip_whitespace();

        while let Some(c) = self.current_char {
            if c.is_alphabetic() && !(c.is_whitespace() || c.is_numeric()) {
                identifier.push(c);
                self.advance();
            } else {
                break;
            }
        }

        if identifier.is_empty() {
            Err(ParseError::InvalidToken("Empty identifier".to_string()))
        } else {
            Ok(identifier)
        }
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();

        while let Some(c) = self.current_char {
            match c {
                ' ' | '\t' | '\n' | '\r' => self.advance(),
                ';' => {
                    tokens.push(self.parse_comment()?);
                    self.advance();
                }
                'c' => {
                    // Lookahead after self.current_char and check if it's a circuit or a cnot
                    let lookahead = self.iter.as_str();
                    if lookahead.starts_with("ircuit") {
                        tokens.push(self.parse_circuit()?);
                    } else {
                        tokens.push(self.parse_gate()?);
                    }
                }
                _ => {
                    tokens.push(self.parse_gate()?);
                    self.advance();
                }
            }
        }

        Ok(tokens)
    }
}

fn read_file(file_path: &str) -> io::Result<String> {
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

fn main() -> Result<(), ParseError> {
    let input: String;
    // Check if a file path is provided as a command-line argument
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: quantuter <file_path>");
        std::process::exit(1);
    }

    let file_path = &args[1];
    // Check if file extension is .quanta
    if !file_path.ends_with(".quanta") {
        eprintln!("File must have .quanta extension");
        std::process::exit(1);
    }

    // Read the file
    match read_file(file_path) {
        Ok(contents) => {
            input = contents;
        }
        Err(err) => {
            eprintln!("Error reading file: {}", err);
            std::process::exit(1);
        }
    }

    // Parse the input
    let mut parser = QuantaParser::new(input.as_str());
    let tokens = parser.tokenize()?;

    // Start creating the circuit based on the tokens
    let mut circuit = QunatumCurcuit::new(2);
    for token in tokens {
        match token {
            Token::Circuit(size, bitstring) => {
                circuit = QunatumCurcuit::new(size);
                // Apply bitstring if provided
                if let Some(bitstring) = bitstring {
                    for (i, c) in bitstring.chars().enumerate() {
                        if c == '1' {
                            // Apply X gate
                            circuit.x(i);
                        }
                    }
                }
                circuit.print_state();
            }
            Token::Comment(_) => {}
            Token::Gate(gate_name, qubits) => match gate_name.as_str() {
                // TODO: Add more gates
                "h" => {
                    circuit.hadamard(qubits[0]);
                    circuit.print_state_at_gate("Hadamard");
                }
                "cnot" => {
                    circuit.cnot(qubits[0], qubits[1]);
                    circuit.print_state_at_gate("CNOT");
                }
                "x" => {
                    circuit.x(qubits[0]);
                    circuit.print_state_at_gate("X");
                }
                "z" => {
                    circuit.z(qubits[0]);
                    circuit.print_state_at_gate("Z");
                }
                "y" => {
                    circuit.y(qubits[0]);
                    circuit.print_state_at_gate("Y");
                }
                "swap" => {
                    circuit.swap(qubits[0], qubits[1]);
                    circuit.print_state_at_gate("SWAP");
                }
                "identity" | "id" => {
                    circuit.identity(qubits[0]);
                    circuit.print_state_at_gate("Identity");
                }
                "fourier" | "qft" => {
                    circuit.qft();
                    circuit.print_state_at_gate("Qauntum Fourier Transform");
                }
                "measure" => {
                    let measure = circuit.measure(qubits[0]);
                    println!("Measured Qubit '{}': {}", qubits[0], measure);
                }
                _ => panic!("Unknown gate `{}`", gate_name),
            },
        }
    }

    Ok(())
}
