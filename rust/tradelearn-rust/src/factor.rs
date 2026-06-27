use pyo3::prelude::*;

#[pyfunction]
pub fn factor_evaluate(
    dates: Vec<i64>,
    factors: Vec<f64>,
    forward_returns: Vec<f64>,
    quantile_labels: Vec<i64>,
    quantiles: usize,
) -> PyResult<(
    Vec<i64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<i64>,
    Vec<f64>,
)> {
    if dates.len() != factors.len()
        || dates.len() != forward_returns.len()
        || dates.len() != quantile_labels.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dates, factors, forward_returns, and quantile_labels must have the same length",
        ));
    }
    if quantiles == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "quantiles must be positive",
        ));
    }

    let mut unique_dates: Vec<i64> = Vec::new();
    let mut ic_values: Vec<f64> = Vec::new();
    let mut rank_ic_values: Vec<f64> = Vec::new();
    let mut quantile_returns: Vec<f64> = Vec::new();
    let mut quantile_counts: Vec<i64> = Vec::new();
    let mut spread_values: Vec<f64> = Vec::new();

    let mut start = 0usize;
    while start < dates.len() {
        let date = dates[start];
        let mut end = start + 1;
        while end < dates.len() && dates[end] == date {
            end += 1;
        }
        let n = end - start;
        let factor_slice = &factors[start..end];
        let return_slice = &forward_returns[start..end];
        let label_slice = &quantile_labels[start..end];

        unique_dates.push(date);
        ic_values.push(pearson_corr(factor_slice, return_slice));
        let factor_ranks = average_ranks(factor_slice);
        let return_ranks = average_ranks(return_slice);
        rank_ic_values.push(pearson_corr(&factor_ranks, &return_ranks));

        let mut sums = vec![0.0; quantiles];
        let mut counts = vec![0i64; quantiles];
        for i in 0..n {
            let label = label_slice[i];
            if label >= 1 && label <= quantiles as i64 {
                let q = (label - 1) as usize;
                sums[q] += return_slice[i];
                counts[q] += 1;
            }
        }
        for q in 0..quantiles {
            quantile_returns.push(if counts[q] > 0 {
                sums[q] / counts[q] as f64
            } else {
                f64::NAN
            });
            quantile_counts.push(counts[q]);
        }
        let bottom = if counts[0] > 0 {
            sums[0] / counts[0] as f64
        } else {
            f64::NAN
        };
        let top_idx = quantiles - 1;
        let top = if counts[top_idx] > 0 {
            sums[top_idx] / counts[top_idx] as f64
        } else {
            f64::NAN
        };
        spread_values.push(top - bottom);
        start = end;
    }

    Ok((
        unique_dates,
        ic_values,
        rank_ic_values,
        quantile_returns,
        quantile_counts,
        spread_values,
    ))
}

pub fn register_pyfunctions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(factor_evaluate, m)?)?;
    Ok(())
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let mut ranks = vec![0.0; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && values[order[j]] == values[order[i]] {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[order[k]] = avg_rank;
        }
        i = j;
    }
    ranks
}

fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 || y.len() != n {
        return f64::NAN;
    }
    let x_mean = x.iter().sum::<f64>() / n as f64;
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;
    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        cov += dx * dy;
        x_var += dx * dx;
        y_var += dy * dy;
    }
    if x_var == 0.0 || y_var == 0.0 {
        return f64::NAN;
    }
    cov / (x_var.sqrt() * y_var.sqrt())
}
