use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use std::f64::consts::PI;
const TAU: f64 = 2.0 * PI;


// Waveform definitions

#[pyfunction]
fn harmonic_oscillator(
    py: Python,
    time: &PyArray1<f64>,
    amplitude: f64,     // A: Ampitude
    gamma: f64,         // gamma: Damping coefficient
    omega: f64,         // omega: Frequency of carrier wave
    phi: f64,           // phi: Phase shift
    y: f64,             // y: equilibrium value
    ) -> Py<PyArray1<f64>>{
        let binding = time.readonly();
        let time_vec = binding.as_slice().unwrap();
        let mut harmonic_result: Vec<f64> = vec![0.0; time_vec.len()];

        for (i, &t) in time_vec.iter().enumerate() {
            let damping_factor = amplitude * (gamma * t).exp();

            let oscillator = (omega * (t + phi)).cos();

            harmonic_result[i] = damping_factor * oscillator + y
            }
    harmonic_result.to_pyarray(py).to_owned()
}

#[pyfunction]
fn fourier_square_wave(
    py: Python,
    time: &PyArray1<f64>,
    amplitude: f64,     // A: Ampitude
    gamma: f64,         // gamma: Damping coefficient
    omega: f64,         // omega: Frequency of carrier wave
    phi: f64,           // phi: Phase shift
    y: f64,             // y: equilibrium value
    num_harmonics: usize,   // Number of harmonics
    ) -> Py<PyArray1<f64>>{
        let binding = time.readonly();
        let time_vec = binding.as_slice().unwrap();
        let mut square_wave: Vec<f64> = vec![0.0; time_vec.len()];

        for (i, &t) in time_vec.iter().enumerate() {
            let damping_factor = amplitude * (gamma * t).exp();
            let mut harmonic_sum = 0.0;

            for k in (1..=(2 * num_harmonics)).step_by(2) {
                let k_f64 = k as f64;
                let freq = k_f64 * omega;
                harmonic_sum += (4.0 / PI) * (freq * (t+phi)).sin() / k_f64;
            }
        square_wave[i] = damping_factor * (1.0 + harmonic_sum) + y;
        }
    square_wave.to_pyarray(py).to_owned()
}

#[pyfunction]
fn pseudo_square_wave(
    py: Python,
    time: &PyArray1<f64>,
    amplitude: f64,     // A: Ampitude
    gamma: f64,         // gamma: Damping coefficient
    omega: f64,         // omega: Frequency of carrier wave
    phi: f64,           // phu: Phase shift
    y: f64,             // y: equilibrium value
    ) -> Py<PyArray1<f64>>{
        let binding = time.readonly();
        let time_vec = binding.as_slice().unwrap();
            let mut square_wave: Vec<f64> = vec![0.0; time_vec.len()];

        for (i, &t) in time_vec.iter().enumerate() {
            let damping_factor = amplitude * (gamma * t).exp();

            square_wave[i] = damping_factor * ((omega * (t+phi)).sin() + 0.5*((omega * (t+phi))*3.0).sin()) + y;
            }
    square_wave.to_pyarray(py).to_owned()
}

#[pyfunction]
fn pseudo_cycloid_wave(
    py: Python,
    time: &PyArray1<f64>,
    amplitude: f64,     // A: Ampitude
    gamma: f64,         // gamma: Damping coefficient
    omega: f64,         // omega: Frequency of carrier wave
    phi: f64,           // phu: Phase shift
    y: f64,             // y: equilibrium value
    ) -> Py<PyArray1<f64>>{
        let binding = time.readonly();
        let time_vec = binding.as_slice().unwrap();
        let mut cycloid_wave: Vec<f64> = vec![0.0; time_vec.len()];

    for (i, &t) in time_vec.iter().enumerate() {
        let damping_factor = amplitude * (gamma * t).exp();

        cycloid_wave[i] = damping_factor * ((omega * 2.0 * (t+phi)).cos() - 2.0*(omega* (t + phi)).cos()) + y;
        }
    cycloid_wave.to_pyarray(py).to_owned()
}

//fn fourier_cycloid_wave

#[pyfunction]
fn transient_impulse(
    py: Python,
    time: &PyArray1<f64>, // t: Time variable
    amplitude: f64,       // A: Amplitude of the pulse
    period: f64,          // p: Period of the pulse center of the pulse
    width: f64,           // w: Pulse width
    equilibrium: f64,     // y: Equilibrium value
    ) -> Py<PyArray1<f64>> {
        let binding = time.readonly();
        let time_vec = binding.as_slice().unwrap();
        let mut result: Vec<f64> = vec![0.0; time_vec.len()];
        let p_tau = (period / 24.0) * (2.0*PI);

        for (i,&t) in time_vec.iter().enumerate() {
            let t_mod = t % (TAU - 0.00000001); // Need to fix this mod has 0 and 24 as distinct, therefore I have
//24-0.000001 in the mod. This introduces a small but accumulating error as the number of cycles in the dataset increases
// Could just have t_mod = 23?

        let impulse = if t_mod - p_tau >= 0.0 {
            (-0.5 * ((t_mod - p_tau) / width).powi(2)).exp()
            } else {0.0};
        result[i] = amplitude * impulse + equilibrium;
    }
    result.to_pyarray(py).to_owned()
}


/// Python module definition
#[pymodule]
fn pycycle_oscillators(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(harmonic_oscillator, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_square_wave, m)?)?;
    m.add_function(wrap_pyfunction!(fourier_square_wave, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_cycloid_wave, m)?)?;
    m.add_function(wrap_pyfunction!(transient_impulse, m)?)?;
    Ok(())
}
