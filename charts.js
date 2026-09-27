/**
 * EDA Website Charts
 * All Chart.js visualizations for the FinTech Alert Escalation EDA report.
 * Data is embedded directly from the EDA extraction pipeline.
 */

// ========================================
// EMBEDDED DATA (from eda_data.json)
// ========================================
const EDA = {
    overview: {
        train_signals: 14000,
        test_signals: 6000,
        train_transactions: 6987663,
        test_transactions: 3027575,
        avg_txn_per_signal: 499.12,
        escalation_rate: 0.1718,
    },
    target_distribution: {
        labels: ['Dismissed (0)', 'Escalated (1)'],
        values: [11595, 2405],
        percentages: [82.82, 17.18],
    },
    type_analysis: {
        type_labels: ['Card', 'Bank Transfer', 'Cash', 'International'],
        dismissed_counts: [3082384, 2259238, 360454, 25920],
        escalated_counts: [676413, 495449, 81892, 5913],
        dismissed_avg_amount: [-0.4247, 0.1605, 0.5323, 2.0054],
        escalated_avg_amount: [-0.4513, 0.0410, 0.5366, 1.9165],
    },
    amount_per_signal: {
        dismissed_mean_avg: -0.0452,
        escalated_mean_avg: -0.1161,
        dismissed_std_avg: 0.8718,
        escalated_std_avg: 0.8390,
        dismissed_max_avg: 3.0521,
        escalated_max_avg: 2.8817,
    },
    txn_count_distribution: {
        dismissed_mean: 494.01,
        escalated_mean: 523.77,
        dismissed_median: 455,
        escalated_median: 493,
        dismissed_hist: [577,762,914,1164,1243,1199,1163,1013,847,716,543,402,318,253,149,116,75,44,33,24,14,11,5,1,2,2,2,2,0,1],
        escalated_hist: [111,122,172,212,223,262,245,229,183,187,121,99,69,50,37,23,19,15,8,6,4,3,1,3,0,0,1,0,0,0],
        hist_bins_count: 30,
    },
    hourly_activity: {
        hours: Array.from({length: 24}, (_, i) => i),
        dismissed_counts: [222764,219889,219682,220200,220435,220508,219405,219184,219770,219963,220459,219120,219504,220013,222117,219185,219872,219720,218702,218621,218985,219526,219176,671196],
        escalated_counts: [48943,48816,48334,48532,48665,48364,48348,48633,48517,48020,48408,48360,48590,48161,48801,48332,48174,48044,48389,48668,48422,48067,48308,145771],
    },
    dow_activity: {
        days: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        dismissed_counts: [819024,818147,820472,814557,821817,819391,814588],
        escalated_counts: [179956,179405,180324,180614,179848,181417,178103],
    },
    monthly_activity: {
        months: ['2024-07','2024-08','2024-09','2024-10','2024-11','2024-12','2025-01','2025-02','2025-03','2025-04','2025-05','2025-06','2025-07','2025-08','2025-09','2025-10','2025-11','2025-12','2026-01','2026-02','2026-03','2026-04','2026-05','2026-06','2026-07','2026-08','2026-09','2026-10','2026-11','2026-12'],
        dismissed_counts: [15471,54629,87324,118617,136981,159826,194424,204096,239577,253627,285778,293999,323105,319586,311836,317853,295175,295697,281257,232551,236267,203516,194434,176215,157351,127605,92764,64028,36546,17861],
        escalated_counts: [3630,11667,20051,27844,31940,36115,41270,43954,49886,51739,57572,58959,64444,63904,64105,70450,66943,68988,64058,54540,55722,46586,41958,38804,37717,31941,24655,16277,8623,5325],
    },
    signal_monthly: {
        months: ['2025-01','2025-02','2025-03','2025-04','2025-05','2025-06','2025-07','2025-08','2025-09','2025-10','2025-11','2025-12','2026-01','2026-02','2026-03','2026-04','2026-05','2026-06','2026-07','2026-08','2026-09','2026-10','2026-11','2026-12'],
        dismissed_counts: [419,421,416,376,389,426,624,610,599,709,668,664,641,591,623,479,527,543,352,344,326,278,265,305],
        escalated_counts: [86,96,89,94,77,71,122,125,97,147,122,141,126,113,155,129,108,91,75,75,69,59,66,72],
    },
    direction_analysis: {
        kirim_avg_amount: [-0.2147, -0.2751],
        chiqim_avg_amount: [0.1647, 0.1004],
    },
    recent_activity: {
        dismissed_30d_mean: 77.9,
        escalated_30d_mean: 81.2,
    },
};

// Model results (updated from Optuna + Top-75 Feature Selection + 5-Seed Ensemble)
const MODEL_RESULTS = {
    lgb: { mean_auc: 0.6465, std_auc: 0.0016 },
    xgb: { mean_auc: 0.6330, std_auc: 0.0092 },
    cat: { mean_auc: 0.6449, std_auc: 0.0018 },
    ensemble_auc: 0.6505,
    top_features: [
        ['chiqim_bank_otkazmasi_mean', 0.0550],
        ['kirim_naqd_max', 0.0474],
        ['karta_min', 0.0436],
        ['bank_otkazmasi_std', 0.0436],
        ['kirim_naqd_mean', 0.0408],
        ['amt_diff_min', 0.0342],
        ['karta_max', 0.0323],
        ['chiqim_karta_min', 0.0304],
        ['bank_otkazmasi_mean', 0.0304],
        ['chiqim_bank_otkazmasi_std', 0.0275],
        ['kirim_min', 0.0266],
        ['kirim_bank_otkazmasi_mean', 0.0266],
        ['chiqim_karta_mean', 0.0237],
        ['chiqim_bank_otkazmasi_min', 0.0209],
        ['bank_otkazmasi_min', 0.0199],
    ],
};

// ========================================
// CHART DEFAULTS
// ========================================
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = '#64748b';
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyle = 'circle';
Chart.defaults.plugins.legend.labels.padding = 16;

const COLORS = {
    dismissed: '#94a3b8',
    dismissedBg: 'rgba(148, 163, 184, 0.15)',
    escalated: '#1a56db',
    escalatedBg: 'rgba(26, 86, 219, 0.15)',
    primary: '#1a56db',
    primaryBg: 'rgba(26, 86, 219, 0.1)',
    accent: '#0ea5e9',
    accentBg: 'rgba(14, 165, 233, 0.1)',
    success: '#10b981',
    successBg: 'rgba(16, 185, 129, 0.1)',
    warning: '#f59e0b',
    warningBg: 'rgba(245, 158, 11, 0.1)',
};

// ========================================
// CHART 1: Target Distribution (Doughnut)
// ========================================
new Chart(document.getElementById('chartTargetDist'), {
    type: 'doughnut',
    data: {
        labels: EDA.target_distribution.labels,
        datasets: [{
            data: EDA.target_distribution.values,
            backgroundColor: [COLORS.dismissed, COLORS.primary],
            borderWidth: 0,
            hoverOffset: 8,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        plugins: {
            legend: {
                position: 'bottom',
                labels: { font: { size: 13 } }
            },
            tooltip: {
                callbacks: {
                    label: (ctx) => {
                        const pct = EDA.target_distribution.percentages[ctx.dataIndex];
                        return `${ctx.label}: ${ctx.raw.toLocaleString()} (${pct.toFixed(1)}%)`;
                    }
                }
            }
        }
    }
});

// ========================================
// CHART 2: Monthly Escalation Rate
// ========================================
const monthlyEscRate = EDA.signal_monthly.escalated_counts.map((esc, i) => {
    const total = esc + EDA.signal_monthly.dismissed_counts[i];
    return total > 0 ? (esc / total * 100) : 0;
});

new Chart(document.getElementById('chartMonthlyEscalation'), {
    type: 'line',
    data: {
        labels: EDA.signal_monthly.months.map(m => {
            const [y, mo] = m.split('-');
            return `${['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][parseInt(mo)-1]} '${y.slice(2)}`;
        }),
        datasets: [{
            label: 'Escalation Rate (%)',
            data: monthlyEscRate,
            borderColor: COLORS.primary,
            backgroundColor: COLORS.primaryBg,
            fill: true,
            tension: 0.4,
            pointRadius: 3,
            pointHoverRadius: 6,
            borderWidth: 2,
        }, {
            label: 'Overall Average (17.2%)',
            data: new Array(monthlyEscRate.length).fill(17.18),
            borderColor: COLORS.dismissed,
            borderDash: [6, 4],
            pointRadius: 0,
            borderWidth: 1.5,
            fill: false,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                max: 35,
                title: { display: true, text: 'Escalation Rate (%)' },
                grid: { color: 'rgba(0,0,0,0.05)' },
            },
            x: {
                grid: { display: false },
                ticks: { maxRotation: 45 }
            }
        },
        plugins: {
            legend: { position: 'top' },
        }
    }
});

// ========================================
// CHART 3: Transaction Count Distribution
// ========================================
const txnBinLabels = [];
for (let i = 0; i < 30; i++) {
    const start = Math.round(i * (2300 / 30));
    const end = Math.round((i + 1) * (2300 / 30));
    txnBinLabels.push(`${start}-${end}`);
}

new Chart(document.getElementById('chartTxnCount'), {
    type: 'bar',
    data: {
        labels: txnBinLabels,
        datasets: [{
            label: 'Dismissed',
            data: EDA.txn_count_distribution.dismissed_hist,
            backgroundColor: COLORS.dismissedBg,
            borderColor: COLORS.dismissed,
            borderWidth: 1,
        }, {
            label: 'Escalated',
            data: EDA.txn_count_distribution.escalated_hist,
            backgroundColor: COLORS.escalatedBg,
            borderColor: COLORS.escalated,
            borderWidth: 1,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                title: { display: true, text: 'Number of Signals' },
                grid: { color: 'rgba(0,0,0,0.05)' },
            },
            x: {
                title: { display: true, text: 'Transaction Count' },
                grid: { display: false },
                ticks: {
                    maxRotation: 45,
                    callback: function(val, index) {
                        return index % 5 === 0 ? this.getLabelForValue(val) : '';
                    }
                }
            }
        },
        plugins: {
            legend: { position: 'top' },
        }
    }
});

// ========================================
// CHART 4: Transaction Types
// ========================================
new Chart(document.getElementById('chartTxnTypes'), {
    type: 'bar',
    data: {
        labels: EDA.type_analysis.type_labels,
        datasets: [{
            label: 'Dismissed',
            data: EDA.type_analysis.dismissed_counts.map(c => c / 1000),
            backgroundColor: COLORS.dismissedBg,
            borderColor: COLORS.dismissed,
            borderWidth: 1.5,
            borderRadius: 4,
        }, {
            label: 'Escalated',
            data: EDA.type_analysis.escalated_counts.map(c => c / 1000),
            backgroundColor: COLORS.escalatedBg,
            borderColor: COLORS.escalated,
            borderWidth: 1.5,
            borderRadius: 4,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                title: { display: true, text: 'Count (thousands)' },
                grid: { color: 'rgba(0,0,0,0.05)' },
            },
            x: {
                grid: { display: false }
            }
        },
        plugins: {
            legend: { position: 'top' },
            tooltip: {
                callbacks: {
                    label: (ctx) => `${ctx.dataset.label}: ${(ctx.raw * 1000).toLocaleString()}`
                }
            }
        }
    }
});

// ========================================
// CHART 5: Amount Distribution
// ========================================
const amountBins = [];
for (let i = 0; i < 50; i++) {
    const val = -3 + i * 0.2;
    amountBins.push(val.toFixed(1));
}

const dismissedAmtHist = [9707,6419,9031,14866,45849,59171,97250,157922,239655,339690,432126,497283,527102,517014,479255,426385,367426,308405,257515,210456,170303,135612,106127,81462,61773,46631,34419,25105,18371,13241,9484,6673,4839,3477,2406,1799,1148,814,574,425,245,169,127,87,53,42,25,24,14,0];
const escalatedAmtHist = [2058,1616,2204,3984,10744,14292,24084,38432,57589,79265,99277,112057,118129,113891,104750,92405,78680,66039,54354,43910,35184,27669,21203,15721,11787,8711,6280,4679,3186,2263,1573,1145,790,522,397,277,176,113,61,67,28,33,14,13,6,3,4,2,0,0];

// Normalize to density for fair comparison
const dismissedTotal = dismissedAmtHist.reduce((a, b) => a + b, 0);
const escalatedTotal = escalatedAmtHist.reduce((a, b) => a + b, 0);

new Chart(document.getElementById('chartAmountDist'), {
    type: 'line',
    data: {
        labels: amountBins,
        datasets: [{
            label: 'Dismissed (density)',
            data: dismissedAmtHist.map(v => v / dismissedTotal * 100),
            borderColor: COLORS.dismissed,
            backgroundColor: COLORS.dismissedBg,
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            borderWidth: 2,
        }, {
            label: 'Escalated (density)',
            data: escalatedAmtHist.map(v => v / escalatedTotal * 100),
            borderColor: COLORS.escalated,
            backgroundColor: COLORS.escalatedBg,
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            borderWidth: 2,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                title: { display: true, text: 'Density (%)' },
                grid: { color: 'rgba(0,0,0,0.05)' },
            },
            x: {
                title: { display: true, text: 'miqdor_indeksi' },
                grid: { display: false },
                ticks: {
                    callback: function(val, index) {
                        return index % 5 === 0 ? this.getLabelForValue(val) : '';
                    }
                }
            }
        },
        plugins: {
            legend: { position: 'top' },
        }
    }
});

// ========================================
// CHART 6: Average Amount by Type
// ========================================
new Chart(document.getElementById('chartTypeAmounts'), {
    type: 'bar',
    data: {
        labels: EDA.type_analysis.type_labels,
        datasets: [{
            label: 'Dismissed',
            data: EDA.type_analysis.dismissed_avg_amount,
            backgroundColor: COLORS.dismissedBg,
            borderColor: COLORS.dismissed,
            borderWidth: 1.5,
            borderRadius: 4,
        }, {
            label: 'Escalated',
            data: EDA.type_analysis.escalated_avg_amount,
            backgroundColor: COLORS.escalatedBg,
            borderColor: COLORS.escalated,
            borderWidth: 1.5,
            borderRadius: 4,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                title: { display: true, text: 'Average miqdor_indeksi' },
                grid: { color: 'rgba(0,0,0,0.05)' },
            },
            x: {
                grid: { display: false }
            }
        },
        plugins: {
            legend: { position: 'top' },
        }
    }
});

// ========================================
// CHART 7: Hourly Distribution
// ========================================
const hourLabels = EDA.hourly_activity.hours.map(h => `${h.toString().padStart(2, '0')}:00`);
const dismissedHourlyNorm = EDA.hourly_activity.dismissed_counts.map(c => c / 1000);
const escalatedHourlyNorm = EDA.hourly_activity.escalated_counts.map(c => c / 1000);

new Chart(document.getElementById('chartHourly'), {
    type: 'bar',
    data: {
        labels: hourLabels,
        datasets: [{
            label: 'Dismissed (K)',
            data: dismissedHourlyNorm,
            backgroundColor: COLORS.dismissedBg,
            borderColor: COLORS.dismissed,
            borderWidth: 1,
            borderRadius: 2,
        }, {
            label: 'Escalated (K)',
            data: escalatedHourlyNorm,
            backgroundColor: COLORS.escalatedBg,
            borderColor: COLORS.escalated,
            borderWidth: 1,
            borderRadius: 2,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                title: { display: true, text: 'Transactions (thousands)' },
                grid: { color: 'rgba(0,0,0,0.05)' },
            },
            x: {
                grid: { display: false },
                ticks: {
                    callback: function(val, index) {
                        return index % 3 === 0 ? this.getLabelForValue(val) : '';
                    }
                }
            }
        },
        plugins: {
            legend: { position: 'top' },
        }
    }
});

// ========================================
// CHART 8: Day of Week
// ========================================
new Chart(document.getElementById('chartDow'), {
    type: 'bar',
    data: {
        labels: EDA.dow_activity.days,
        datasets: [{
            label: 'Dismissed',
            data: EDA.dow_activity.dismissed_counts.map(c => c / 1000),
            backgroundColor: COLORS.dismissedBg,
            borderColor: COLORS.dismissed,
            borderWidth: 1.5,
            borderRadius: 4,
        }, {
            label: 'Escalated',
            data: EDA.dow_activity.escalated_counts.map(c => c / 1000),
            backgroundColor: COLORS.escalatedBg,
            borderColor: COLORS.escalated,
            borderWidth: 1.5,
            borderRadius: 4,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                title: { display: true, text: 'Transactions (thousands)' },
                grid: { color: 'rgba(0,0,0,0.05)' },
            },
            x: {
                grid: { display: false }
            }
        },
        plugins: {
            legend: { position: 'top' },
        }
    }
});

// ========================================
// CHART 9: Monthly Transaction Volume
// ========================================
new Chart(document.getElementById('chartMonthly'), {
    type: 'line',
    data: {
        labels: EDA.monthly_activity.months.map(m => {
            const [y, mo] = m.split('-');
            return `${['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][parseInt(mo)-1]} '${y.slice(2)}`;
        }),
        datasets: [{
            label: 'Dismissed',
            data: EDA.monthly_activity.dismissed_counts.map(c => c / 1000),
            borderColor: COLORS.dismissed,
            backgroundColor: COLORS.dismissedBg,
            fill: true,
            tension: 0.4,
            pointRadius: 3,
            borderWidth: 2,
        }, {
            label: 'Escalated',
            data: EDA.monthly_activity.escalated_counts.map(c => c / 1000),
            borderColor: COLORS.escalated,
            backgroundColor: COLORS.escalatedBg,
            fill: true,
            tension: 0.4,
            pointRadius: 3,
            borderWidth: 2,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                title: { display: true, text: 'Transactions (thousands)' },
                grid: { color: 'rgba(0,0,0,0.05)' },
            },
            x: {
                grid: { display: false },
                ticks: {
                    maxRotation: 45,
                    callback: function(val, index) {
                        return index % 3 === 0 ? this.getLabelForValue(val) : '';
                    }
                }
            }
        },
        plugins: {
            legend: { position: 'top' },
        }
    }
});

// ========================================
// CHART 10: Direction Ratio Comparison
// ========================================
new Chart(document.getElementById('chartDirectionRatio'), {
    type: 'bar',
    data: {
        labels: ['Dismissed', 'Escalated'],
        datasets: [{
            label: 'Incoming (Kirim) %',
            data: [75.03, 74.41],
            backgroundColor: [COLORS.accentBg, COLORS.accentBg],
            borderColor: [COLORS.accent, COLORS.accent],
            borderWidth: 1.5,
            borderRadius: 4,
        }, {
            label: 'Outgoing (Chiqim) %',
            data: [24.97, 25.59],
            backgroundColor: [COLORS.warningBg, COLORS.warningBg],
            borderColor: [COLORS.warning, COLORS.warning],
            borderWidth: 1.5,
            borderRadius: 4,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        scales: {
            x: {
                stacked: true,
                max: 100,
                title: { display: true, text: 'Percentage' },
                grid: { color: 'rgba(0,0,0,0.05)' },
            },
            y: {
                stacked: true,
                grid: { display: false },
            }
        },
        plugins: {
            legend: { position: 'top' },
            tooltip: {
                callbacks: {
                    label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%`
                }
            }
        }
    }
});

// ========================================
// CHART 11: Direction × Amount Analysis
// ========================================
new Chart(document.getElementById('chartDirectionAnalysis'), {
    type: 'bar',
    data: {
        labels: ['Kirim (Incoming)', 'Chiqim (Outgoing)'],
        datasets: [{
            label: 'Dismissed',
            data: [
                EDA.direction_analysis.kirim_avg_amount[0],
                EDA.direction_analysis.chiqim_avg_amount[0],
            ],
            backgroundColor: COLORS.dismissedBg,
            borderColor: COLORS.dismissed,
            borderWidth: 1.5,
            borderRadius: 4,
        }, {
            label: 'Escalated',
            data: [
                EDA.direction_analysis.kirim_avg_amount[1],
                EDA.direction_analysis.chiqim_avg_amount[1],
            ],
            backgroundColor: COLORS.escalatedBg,
            borderColor: COLORS.escalated,
            borderWidth: 1.5,
            borderRadius: 4,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                title: { display: true, text: 'Average miqdor_indeksi' },
                grid: { color: 'rgba(0,0,0,0.05)' },
            },
            x: {
                grid: { display: false }
            }
        },
        plugins: {
            legend: { position: 'top' },
        }
    }
});

// ========================================
// CHART 12: Feature Importance
// ========================================
const topFeatures = MODEL_RESULTS.top_features.slice().reverse();
const featureLabels = topFeatures.map(f => f[0].replace(/_/g, ' '));
const featureValues = topFeatures.map(f => (f[1] * 100).toFixed(2));

new Chart(document.getElementById('chartFeatureImportance'), {
    type: 'bar',
    data: {
        labels: featureLabels,
        datasets: [{
            label: 'Importance (%)',
            data: featureValues,
            backgroundColor: featureValues.map((_, i) => {
                const ratio = i / featureValues.length;
                return `rgba(26, 86, 219, ${0.3 + ratio * 0.6})`;
            }),
            borderColor: COLORS.primary,
            borderWidth: 1,
            borderRadius: 3,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        scales: {
            x: {
                title: { display: true, text: 'Relative Importance (%)' },
                grid: { color: 'rgba(0,0,0,0.05)' },
            },
            y: {
                grid: { display: false },
                ticks: {
                    font: { family: "'JetBrains Mono', monospace", size: 11 }
                }
            }
        },
        plugins: {
            legend: { display: false },
        }
    }
});

// ========================================
// Update Model Score Cards
// ========================================
document.getElementById('scoreLgb').textContent = MODEL_RESULTS.lgb.mean_auc.toFixed(4);
document.getElementById('scoreXgb').textContent = MODEL_RESULTS.xgb.mean_auc.toFixed(4);
document.getElementById('scoreCat').textContent = MODEL_RESULTS.cat.mean_auc.toFixed(4);

// Weighted ensemble score
const ensembleScore = MODEL_RESULTS.ensemble_auc;
document.getElementById('scoreEnsemble').textContent = ensembleScore.toFixed(4);

// ========================================
// Sticky Nav Active State
// ========================================
const sections = document.querySelectorAll('.section');
const navLinks = document.querySelectorAll('.nav-link');

const observerOptions = {
    root: null,
    rootMargin: '-80px 0px -60% 0px',
    threshold: 0,
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            navLinks.forEach(link => link.classList.remove('active'));
            const activeLink = document.querySelector(`.nav-link[href="#${entry.target.id}"]`);
            if (activeLink) activeLink.classList.add('active');
        }
    });
}, observerOptions);

sections.forEach(section => observer.observe(section));
