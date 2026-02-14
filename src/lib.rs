use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use flexi_logger::{Logger, WriteMode};

use gpredomics::param::Param as GParam;
use gpredomics::data::Data as GData;
use gpredomics::individual::Individual as GIndividual;
use gpredomics::population::Population as GPopulation;
use gpredomics::experiment::Experiment as GExperiment;

// ---------------------------------------------------------------------------
// Helpers: convert u8 language/data_type codes to human-readable strings
// ---------------------------------------------------------------------------

fn language_name(lang: u8) -> &'static str {
    match lang {
        0 => "binary",
        1 => "ternary",
        2 => "pow2",
        3 => "ratio",
        101 => "mcmc_generic",
        _ => "unknown",
    }
}

fn data_type_name(dt: u8) -> &'static str {
    match dt {
        0 => "raw",
        1 => "prevalence",
        2 => "log",
        _ => "unknown",
    }
}

// ---------------------------------------------------------------------------
// Param wrapper
// ---------------------------------------------------------------------------

/// Parameter configuration for gpredomics algorithms.
///
/// Load from a YAML file or create with defaults, then modify individual
/// parameters before passing to `fit()`.
#[pyclass]
#[derive(Clone)]
struct Param {
    inner: GParam,
}

#[pymethods]
impl Param {
    /// Create a new Param with default values.
    #[new]
    fn new() -> Self {
        Param {
            inner: GParam::default(),
        }
    }

    /// Load parameters from a YAML configuration file.
    ///
    /// Args:
    ///     path: Path to the YAML parameter file.
    fn load(&mut self, path: &str) -> PyResult<()> {
        self.inner = gpredomics::param::get(path.to_string())
            .map_err(|e| PyValueError::new_err(format!("Failed to load params: {}", e)))?;
        Ok(())
    }

    /// Set a numeric parameter by name.
    ///
    /// Args:
    ///     name: Parameter name (e.g., "max_epochs", "population_size").
    ///     value: Numeric value to set.
    fn set(&mut self, name: &str, value: f64) -> PyResult<()> {
        match name {
            // General
            "seed" => self.inner.general.seed = value as u64,
            "thread_number" => self.inner.general.thread_number = value as usize,
            "k_penalty" => self.inner.general.k_penalty = value,
            "fr_penalty" => self.inner.general.fr_penalty = value,
            "bias_penalty" => self.inner.general.bias_penalty = value,
            "epsilon" | "data_type_epsilon" => self.inner.general.data_type_epsilon = value,
            "n_model_to_display" => self.inner.general.n_model_to_display = value as u32,

            // GA
            "population_size" => self.inner.ga.population_size = value as u32,
            "max_epochs" => self.inner.ga.max_epochs = value as usize,
            "min_epochs" => self.inner.ga.min_epochs = value as usize,
            "max_age_best_model" => self.inner.ga.max_age_best_model = value as usize,
            "k_min" | "ga_kmin" => self.inner.ga.k_min = value as usize,
            "k_max" | "ga_kmax" => self.inner.ga.k_max = value as usize,
            "select_elite_pct" => self.inner.ga.select_elite_pct = value,
            "select_niche_pct" => self.inner.ga.select_niche_pct = value,
            "select_random_pct" => self.inner.ga.select_random_pct = value,
            "mutated_children_pct" => self.inner.ga.mutated_children_pct = value,

            // Beam
            "best_models_criterion" => self.inner.beam.best_models_criterion = value,
            "max_nb_of_models" => self.inner.beam.max_nb_of_models = value as usize,

            // CV
            "outer_folds" => self.inner.cv.outer_folds = value as usize,
            "inner_folds" => self.inner.cv.inner_folds = value as usize,
            "overfit_penalty" => self.inner.cv.overfit_penalty = value,
            "cv_best_models_ci_alpha" => self.inner.cv.cv_best_models_ci_alpha = value,

            // MCMC
            "n_iter" => self.inner.mcmc.n_iter = value as usize,
            "n_burn" => self.inner.mcmc.n_burn = value as usize,
            "lambda" => self.inner.mcmc.lambda = value,

            // Importance
            "n_permutations_mda" => self.inner.importance.n_permutations_mda = value as usize,

            // Voting
            "min_perf" => self.inner.voting.min_perf = value,
            "min_diversity" => self.inner.voting.min_diversity = value,
            "fbm_ci_alpha" => self.inner.voting.fbm_ci_alpha = value,
            "method_threshold" => self.inner.voting.method_threshold = value,

            _ => return Err(PyValueError::new_err(format!("Unknown numeric parameter: {}", name))),
        }
        Ok(())
    }

    /// Set a string parameter by name.
    ///
    /// Args:
    ///     name: Parameter name (e.g., "algo", "language", "fit").
    ///     value: String value to set.
    fn set_string(&mut self, name: &str, value: &str) -> PyResult<()> {
        match name {
            "algo" => self.inner.general.algo = value.to_string(),
            "language" => self.inner.general.language = value.to_string(),
            "data_type" => self.inner.general.data_type = value.to_string(),
            "fit" => self.inner.general.fit = serde_yaml::from_str(value)
                .map_err(|e| PyValueError::new_err(format!("Invalid fit function: {}", e)))?,
            "log_level" => self.inner.general.log_level = value.to_string(),
            "feature_selection_method" => self.inner.data.feature_selection_method = serde_yaml::from_str(value)
                .map_err(|e| PyValueError::new_err(format!("Invalid feature_selection_method: {}", e)))?,
            _ => return Err(PyValueError::new_err(format!("Unknown string parameter: {}", name))),
        }
        Ok(())
    }

    /// Set a boolean parameter by name.
    fn set_bool(&mut self, name: &str, value: bool) -> PyResult<()> {
        match name {
            "gpu" => self.inner.general.gpu = value,
            "cv" => self.inner.general.cv = value,
            "keep_trace" => self.inner.general.keep_trace = value,
            "features_in_rows" => self.inner.data.features_in_rows = value,
            "inverse_classes" => self.inner.data.inverse_classes = value,
            "vote" => self.inner.voting.vote = value,
            "compute_importance" => self.inner.importance.compute_importance = value,
            "scaled_importance" => self.inner.importance.scaled_importance = value,
            _ => return Err(PyValueError::new_err(format!("Unknown boolean parameter: {}", name))),
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("Param(algo={}, language={}, data_type={})",
            self.inner.general.algo,
            self.inner.general.language,
            self.inner.general.data_type)
    }
}

// ---------------------------------------------------------------------------
// Individual wrapper
// ---------------------------------------------------------------------------

/// A single predictive model discovered by gpredomics.
///
/// Contains the model's features, coefficients, and performance metrics.
#[pyclass]
#[derive(Clone)]
struct Individual {
    inner: GIndividual,
}

#[pymethods]
impl Individual {
    /// Get the model's performance metrics as a dictionary.
    ///
    /// Returns:
    ///     dict with keys: auc, fit, accuracy, sensitivity, specificity,
    ///     threshold, k (number of features), language, data_type, epoch.
    fn get_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("auc", self.inner.auc)?;
        dict.set_item("fit", self.inner.fit)?;
        dict.set_item("accuracy", self.inner.accuracy)?;
        dict.set_item("sensitivity", self.inner.sensitivity)?;
        dict.set_item("specificity", self.inner.specificity)?;
        dict.set_item("threshold", self.inner.threshold)?;
        dict.set_item("k", self.inner.k)?;
        dict.set_item("language", language_name(self.inner.language))?;
        dict.set_item("data_type", data_type_name(self.inner.data_type))?;
        dict.set_item("epoch", self.inner.epoch)?;
        Ok(dict)
    }

    /// Get the model's feature indices and their coefficients.
    ///
    /// Returns:
    ///     dict mapping feature index (int) to coefficient sign (int: -1, 0, or 1).
    fn get_features(&self) -> HashMap<usize, i8> {
        self.inner.features.clone()
    }

    fn __repr__(&self) -> String {
        format!("Individual(k={}, auc={:.4}, fit={:.4}, lang={}, dtype={})",
            self.inner.k, self.inner.auc, self.inner.fit,
            language_name(self.inner.language), data_type_name(self.inner.data_type))
    }
}

// ---------------------------------------------------------------------------
// Population wrapper
// ---------------------------------------------------------------------------

/// A collection of Individual models.
///
/// Populations are produced by each generation of the genetic algorithm
/// or each k-level of beam search.
#[pyclass]
#[derive(Clone)]
struct Population {
    inner: GPopulation,
}

#[pymethods]
impl Population {
    /// Get the number of individuals in this population.
    fn __len__(&self) -> usize {
        self.inner.individuals.len()
    }

    /// Get an individual by index.
    fn get_individual(&self, index: usize) -> PyResult<Individual> {
        if index >= self.inner.individuals.len() {
            return Err(PyValueError::new_err(format!(
                "Index {} out of range (population size: {})",
                index, self.inner.individuals.len()
            )));
        }
        Ok(Individual {
            inner: self.inner.individuals[index].clone(),
        })
    }

    /// Get the best individual (first in sorted population).
    fn best(&self) -> PyResult<Individual> {
        self.get_individual(0)
    }

    fn __repr__(&self) -> String {
        format!("Population(size={})", self.inner.individuals.len())
    }
}

// ---------------------------------------------------------------------------
// Experiment wrapper
// ---------------------------------------------------------------------------

/// Complete experiment results from a gpredomics run.
///
/// Contains all populations (generations), the final best population,
/// training/test data references, and optionally cross-validation results.
/// Collections are organized as folds x generations (1 fold for non-CV runs).
#[pyclass]
struct Experiment {
    inner: GExperiment,
}

#[pymethods]
impl Experiment {
    /// Get the number of folds (1 for non-CV runs).
    fn fold_count(&self) -> usize {
        self.inner.collections.len()
    }

    /// Get the number of generations in a given fold (default: fold 0).
    #[pyo3(signature = (fold=0))]
    fn generation_count(&self, fold: usize) -> PyResult<usize> {
        if fold >= self.inner.collections.len() {
            return Err(PyValueError::new_err(format!(
                "Fold {} out of range (total: {})",
                fold, self.inner.collections.len()
            )));
        }
        Ok(self.inner.collections[fold].len())
    }

    /// Get the final (best) population.
    fn best_population(&self) -> PyResult<Population> {
        match &self.inner.final_population {
            Some(pop) => Ok(Population { inner: pop.clone() }),
            None => Err(PyValueError::new_err("No final population available")),
        }
    }

    /// Get the population from a specific generation and fold.
    ///
    /// Args:
    ///     generation: 0-based generation index.
    ///     fold: 0-based fold index (default 0, use for CV runs).
    #[pyo3(signature = (generation, fold=0))]
    fn get_population(&self, generation: usize, fold: usize) -> PyResult<Population> {
        if fold >= self.inner.collections.len() {
            return Err(PyValueError::new_err(format!(
                "Fold {} out of range (total: {})",
                fold, self.inner.collections.len()
            )));
        }
        if generation >= self.inner.collections[fold].len() {
            return Err(PyValueError::new_err(format!(
                "Generation {} out of range (total: {})",
                generation, self.inner.collections[fold].len()
            )));
        }
        Ok(Population {
            inner: self.inner.collections[fold][generation].clone(),
        })
    }

    /// Get total execution time in seconds.
    fn execution_time(&self) -> f64 {
        self.inner.execution_time
    }

    /// Get the feature names from the training data.
    fn feature_names(&self) -> Vec<String> {
        self.inner.train_data.features.clone()
    }

    /// Get the sample names from the training data.
    fn sample_names(&self) -> Vec<String> {
        self.inner.train_data.samples.clone()
    }

    /// Get the full experiment results display including population, importance, and voting/jury.
    ///
    /// Returns:
    ///     String with the formatted experiment results (same output as the CLI).
    fn display_results(&self) -> String {
        self.inner.display_results()
    }

    fn __repr__(&self) -> String {
        let gens = if !self.inner.collections.is_empty() {
            self.inner.collections[0].len()
        } else {
            0
        };
        format!("Experiment(folds={}, generations={}, time={:.2}s, features={}, samples={})",
            self.inner.collections.len(),
            gens,
            self.inner.execution_time,
            self.inner.train_data.feature_len,
            self.inner.train_data.sample_len)
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Run gpredomics with the given parameters (loads data from param file paths).
///
/// Args:
///     param: Param object with configuration and data file paths.
///
/// Returns:
///     Experiment object containing all results.
#[pyfunction]
fn fit(param: &Param) -> PyResult<Experiment> {
    let running = Arc::new(AtomicBool::new(true));
    let inner = gpredomics::run(&param.inner, running);
    Ok(Experiment { inner })
}

/// Initialize the Rust logger so gpredomics progress output is visible.
///
/// Must be called once before fit(). Subsequent calls are silently ignored.
/// Output goes to stderr, which can be captured by the calling process.
///
/// Args:
///     level: Log level string (default: "info"). Options: "error", "warn", "info", "debug", "trace".
#[pyfunction]
#[pyo3(signature = (level="info"))]
fn init_logger(level: &str) -> PyResult<()> {
    // flexi_logger returns Err if already initialized — silently ignore that.
    let _ = Logger::try_with_str(level)
        .map_err(|e| PyValueError::new_err(format!("Logger config error: {}", e)))?
        .write_mode(WriteMode::Direct)
        .start();
    Ok(())
}

/// Run feature filtering/evaluation without a full optimization run.
///
/// Loads data from file paths specified in the Param object, runs the Rust
/// feature evaluation pipeline (wilcoxon/ttest/bayesian_fisher + FDR correction),
/// and computes per-feature statistics (prevalence, mean, std per class).
///
/// Args:
///     param: Param object with data file paths and filtering configuration set.
///
/// Returns:
///     dict with keys: n_features, n_samples, n_classes, class_labels, class_counts,
///     feature_names, features (list of per-feature stat dicts), selected_count, method.
#[pyfunction]
fn filter_features<'py>(py: Python<'py>, param: &Param) -> PyResult<Bound<'py, PyDict>> {
    // Load data
    let mut data = GData::new();
    data.load_data(
        &param.inner.data.X,
        &param.inner.data.y,
        param.inner.data.features_in_rows,
    ).map_err(|e| PyValueError::new_err(format!("Failed to load data: {}", e)))?;

    // Run feature evaluation (includes statistical testing + FDR correction)
    let (class_0_features, class_1_features) = data.evaluate_features(&param.inner);

    // Build lookup: feature_index -> (class, significance)
    let mut eval_map: HashMap<usize, (u8, f64)> = HashMap::new();
    for &(idx, class, sig) in class_0_features.iter().chain(class_1_features.iter()) {
        eval_map.insert(idx, (class, sig));
    }
    let selected_count = eval_map.len();

    // Compute per-feature stats from sparse X matrix
    let features_list = PyList::empty_bound(py);
    for j in 0..data.feature_len {
        let feat_dict = PyDict::new_bound(py);
        feat_dict.set_item("index", j)?;
        feat_dict.set_item("name", &data.features[j])?;

        // Separate values by class and compute stats
        let mut sum_0: f64 = 0.0;
        let mut sum_1: f64 = 0.0;
        let mut sum_sq_0: f64 = 0.0;
        let mut sum_sq_1: f64 = 0.0;
        let mut count_nz_0: usize = 0;
        let mut count_nz_1: usize = 0;
        let mut n_0: usize = 0;
        let mut n_1: usize = 0;

        for i in 0..data.sample_len {
            let val = data.X.get(&(i, j)).copied().unwrap_or(0.0);
            if data.y[i] == 0 {
                n_0 += 1;
                sum_0 += val;
                sum_sq_0 += val * val;
                if val != 0.0 { count_nz_0 += 1; }
            } else {
                n_1 += 1;
                sum_1 += val;
                sum_sq_1 += val * val;
                if val != 0.0 { count_nz_1 += 1; }
            }
        }

        let mean_0 = if n_0 > 0 { sum_0 / n_0 as f64 } else { 0.0 };
        let mean_1 = if n_1 > 0 { sum_1 / n_1 as f64 } else { 0.0 };
        let std_0 = if n_0 > 1 {
            ((sum_sq_0 / n_0 as f64) - mean_0 * mean_0).max(0.0).sqrt()
        } else { 0.0 };
        let std_1 = if n_1 > 1 {
            ((sum_sq_1 / n_1 as f64) - mean_1 * mean_1).max(0.0).sqrt()
        } else { 0.0 };
        let prev_0 = if n_0 > 0 { count_nz_0 as f64 / n_0 as f64 * 100.0 } else { 0.0 };
        let prev_1 = if n_1 > 0 { count_nz_1 as f64 / n_1 as f64 * 100.0 } else { 0.0 };

        // Overall stats
        let n_total = n_0 + n_1;
        let mean_all = if n_total > 0 { (sum_0 + sum_1) / n_total as f64 } else { 0.0 };
        let sum_sq_all = sum_sq_0 + sum_sq_1;
        let std_all = if n_total > 1 {
            ((sum_sq_all / n_total as f64) - mean_all * mean_all).max(0.0).sqrt()
        } else { 0.0 };
        let prev_all = if n_total > 0 {
            (count_nz_0 + count_nz_1) as f64 / n_total as f64 * 100.0
        } else { 0.0 };

        feat_dict.set_item("mean", mean_all)?;
        feat_dict.set_item("std", std_all)?;
        feat_dict.set_item("prevalence", prev_all)?;
        feat_dict.set_item("mean_0", mean_0)?;
        feat_dict.set_item("mean_1", mean_1)?;
        feat_dict.set_item("std_0", std_0)?;
        feat_dict.set_item("std_1", std_1)?;
        feat_dict.set_item("prevalence_0", prev_0)?;
        feat_dict.set_item("prevalence_1", prev_1)?;

        // Evaluation results (class assignment + significance)
        if let Some(&(class, sig)) = eval_map.get(&j) {
            feat_dict.set_item("selected", true)?;
            feat_dict.set_item("class", class)?;
            feat_dict.set_item("significance", sig)?;
        } else {
            feat_dict.set_item("selected", false)?;
            feat_dict.set_item("class", 2u8)?;  // not significant
            feat_dict.set_item("significance", py.None())?;
        }

        features_list.append(feat_dict)?;
    }

    // Build class counts
    let class_counts = PyDict::new_bound(py);
    let mut c0: usize = 0;
    let mut c1: usize = 0;
    for &y_val in &data.y {
        if y_val == 0 { c0 += 1; } else { c1 += 1; }
    }
    if data.classes.len() >= 2 {
        class_counts.set_item(&data.classes[0], c0)?;
        class_counts.set_item(&data.classes[1], c1)?;
    } else {
        class_counts.set_item("0", c0)?;
        class_counts.set_item("1", c1)?;
    }

    // Method name
    let method_name = format!("{:?}", param.inner.data.feature_selection_method);

    // Build result
    let result = PyDict::new_bound(py);
    result.set_item("n_features", data.feature_len)?;
    result.set_item("n_samples", data.sample_len)?;
    result.set_item("n_classes", data.classes.len())?;
    result.set_item("class_labels", &data.classes)?;
    result.set_item("class_counts", class_counts)?;
    result.set_item("feature_names", &data.features)?;
    result.set_item("features", features_list)?;
    result.set_item("selected_count", selected_count)?;
    result.set_item("method", method_name)?;

    Ok(result)
}

/// Python module for gpredomicspy
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Param>()?;
    m.add_class::<Individual>()?;
    m.add_class::<Population>()?;
    m.add_class::<Experiment>()?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(filter_features, m)?)?;
    m.add_function(wrap_pyfunction!(init_logger, m)?)?;
    Ok(())
}
