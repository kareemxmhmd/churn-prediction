"use client";

import React, { useState, useRef, useMemo } from "react";
import {
  UploadCloud,
  FileSpreadsheet,
  Download,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Search,
  Filter,
  Users,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Trash2,
  Check,
} from "lucide-react";
import {
  parseCSV,
  mapRowToCustomer,
  exportToCSV,
  generateSampleCSV,
  BatchItem,
  PredictionResult,
} from "@/lib/csv-parser";

interface BatchPredictionProps {
  apiUrl: string;
}

export default function BatchPrediction({ apiUrl }: BatchPredictionProps) {
  const [file, setFile] = useState<File | null>(null);
  const [items, setItems] = useState<BatchItem[]>([]);
  const [isParsing, setIsParsing] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [error, setError] = useState<string>("");
  const [hasPredicted, setHasPredicted] = useState(false);

  // Table filters & pagination
  const [searchQuery, setSearchQuery] = useState("");
  const [riskFilter, setRiskFilter] = useState<string>("ALL");
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 15;

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (selectedFile: File) => {
    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setError("Please select a valid .csv file.");
      return;
    }

    setError("");
    setFile(selectedFile);
    setIsParsing(true);
    setHasPredicted(false);
    setCurrentPage(1);

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const { rows } = parseCSV(text);

        if (rows.length === 0) {
          setError("The selected CSV file appears to be empty.");
          setItems([]);
          setIsParsing(false);
          return;
        }

        const mappedItems = rows.map((row, idx) => mapRowToCustomer(row, idx));
        setItems(mappedItems);
      } catch (err: any) {
        setError(`Failed to read CSV: ${err.message}`);
      } finally {
        setIsParsing(false);
      }
    };
    reader.onerror = () => {
      setError("Failed to read file.");
      setIsParsing(false);
    };
    reader.readAsText(selectedFile);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    setFile(null);
    setItems([]);
    setError("");
    setHasPredicted(false);
    setProgress({ current: 0, total: 0 });
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const downloadTemplate = () => {
    const csvContent = generateSampleCSV();
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "sample_churn_prediction_template.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const runBatchPredictions = async () => {
    if (items.length === 0) return;

    setIsPredicting(true);
    setError("");
    setProgress({ current: 0, total: items.length });

    try {
      // First attempt fast vectorized batch endpoint
      const batchPayload = items.map((it) => it.customer);
      let batchSuccess = false;
      let batchResults: PredictionResult[] = [];

      try {
        const res = await fetch(`${apiUrl}/predict-batch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(batchPayload),
        });

        if (res.ok) {
          const data = await res.json();
          if (data.results && Array.isArray(data.results)) {
            batchResults = data.results;
            batchSuccess = true;
          }
        }
      } catch (err) {
        console.warn("predict-batch failed, falling back to chunked predict requests", err);
      }

      if (batchSuccess) {
        setItems((prev) =>
          prev.map((item, idx) => ({
            ...item,
            prediction: batchResults[idx],
          }))
        );
        setProgress({ current: items.length, total: items.length });
        setHasPredicted(true);
      } else {
        // Fallback: chunked requests to standard /predict
        const updated = [...items];
        const chunkSize = 5;
        let processedCount = 0;

        for (let i = 0; i < updated.length; i += chunkSize) {
          const chunk = updated.slice(i, i + chunkSize);
          await Promise.all(
            chunk.map(async (item, chunkIdx) => {
              const actualIdx = i + chunkIdx;
              try {
                const res = await fetch(`${apiUrl}/predict`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(item.customer),
                });
                if (!res.ok) throw new Error("Status " + res.status);
                const predData = await res.json();
                updated[actualIdx] = {
                  ...updated[actualIdx],
                  prediction: predData,
                };
              } catch (err: any) {
                updated[actualIdx] = {
                  ...updated[actualIdx],
                  error: err.message || "Failed",
                };
              }
            })
          );
          processedCount += chunk.length;
          setProgress({ current: Math.min(processedCount, items.length), total: items.length });
        }

        setItems(updated);
        setHasPredicted(true);
      }
    } catch (err: any) {
      setError(`Batch prediction encountered an error: ${err.message}`);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleExport = () => {
    if (items.length === 0) return;

    const exportRows = items.map((it) => {
      const pred = it.prediction;
      return {
        ...it.raw,
        Prediction_Churn: pred ? (pred.churn_prediction ? "Will Churn" : "Will Stay") : "N/A",
        Churn_Probability_Pct: pred ? (pred.churn_probability * 100).toFixed(1) + "%" : "N/A",
        Risk_Tier: pred ? pred.risk_tier : "N/A",
      };
    });

    const outName = file ? file.name.replace(/\.csv$/i, "") + "_predictions.csv" : "churn_predictions.csv";
    exportToCSV(exportRows, outName);
  };

  // Stats calculation
  const stats = useMemo(() => {
    if (!hasPredicted) return null;
    const total = items.length;
    const predictedItems = items.filter((it) => it.prediction);
    const churnCount = predictedItems.filter((it) => it.prediction?.churn_prediction).length;
    const highRisk = predictedItems.filter((it) => it.prediction?.risk_tier === "High").length;
    const medRisk = predictedItems.filter((it) => it.prediction?.risk_tier === "Medium").length;
    const lowRisk = predictedItems.filter((it) => it.prediction?.risk_tier === "Low").length;
    const churnRate = total > 0 ? (churnCount / total) * 100 : 0;

    return { total, churnCount, churnRate, highRisk, medRisk, lowRisk };
  }, [items, hasPredicted]);

  // Filtered & paginated items
  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      // Risk filter
      if (riskFilter !== "ALL") {
        if (riskFilter === "CHURN" && !item.prediction?.churn_prediction) return false;
        if (riskFilter === "STAY" && item.prediction?.churn_prediction) return false;
        if (["High", "Medium", "Low"].includes(riskFilter) && item.prediction?.risk_tier !== riskFilter) {
          return false;
        }
      }

      // Search query filter
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const idMatch = String(item.id).toLowerCase().includes(q);
        const genderMatch = item.customer.Gender.toLowerCase().includes(q);
        const deviceMatch = item.customer.PreferredLoginDevice.toLowerCase().includes(q);
        const catMatch = item.customer.PreferedOrderCat.toLowerCase().includes(q);
        const payMatch = item.customer.PreferredPaymentMode.toLowerCase().includes(q);
        return idMatch || genderMatch || deviceMatch || catMatch || payMatch;
      }

      return true;
    });
  }, [items, riskFilter, searchQuery]);

  const totalPages = Math.ceil(filteredItems.length / pageSize) || 1;
  const paginatedItems = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredItems.slice(start, start + pageSize);
  }, [filteredItems, currentPage]);

  return (
    <div className="space-y-6">
      {/* Upload & Template Header Card */}
      <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-gray-100">
          <div>
            <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
              <FileSpreadsheet className="w-5 h-5 text-blue-600" />
              Batch Customer Churn Prediction
            </h2>
            <p className="text-sm text-gray-500 mt-1">
              Upload a CSV file of customer data to predict churn probabilities in bulk.
            </p>
          </div>
          <button
            onClick={downloadTemplate}
            type="button"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-gray-300 bg-white text-sm font-medium text-gray-700 hover:bg-gray-50 shadow-sm transition-colors cursor-pointer"
          >
            <Download className="w-4 h-4 text-gray-500" />
            Download Sample CSV
          </button>
        </div>

        {/* Drag & Drop Area */}
        <div className="mt-6">
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
            accept=".csv"
            className="hidden"
          />

          {!file ? (
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-8 sm:p-12 text-center cursor-pointer transition-all ${
                isDragging
                  ? "border-blue-500 bg-blue-50/50 scale-[0.99]"
                  : "border-gray-300 hover:border-blue-400 bg-gray-50/50 hover:bg-gray-50"
              }`}
            >
              <div className="mx-auto w-14 h-14 rounded-full bg-blue-50 flex items-center justify-center mb-4 text-blue-600">
                <UploadCloud className="w-7 h-7" />
              </div>
              <p className="text-base font-semibold text-gray-800">
                Click to upload or drag & drop customer CSV
              </p>
              <p className="text-xs text-gray-500 mt-1.5 max-w-sm mx-auto">
                Supports any standard .csv file containing customer records (e.g., Tenure, CityTier, WarehouseToHome, etc.)
              </p>
              <div className="mt-4 inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-100/60 text-blue-700 text-xs font-medium">
                Tip: Missing columns or values will automatically use sensible defaults
              </div>
            </div>
          ) : (
            <div className="bg-blue-50/60 border border-blue-200/80 rounded-xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-blue-600 text-white flex items-center justify-center flex-shrink-0">
                  <FileSpreadsheet className="w-5 h-5" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-gray-900 text-sm">{file.name}</span>
                    <span className="px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 text-xs font-medium">
                      {(file.size / 1024).toFixed(1)} KB
                    </span>
                  </div>
                  <p className="text-xs text-gray-600 mt-0.5">
                    {isParsing ? "Reading CSV..." : `${items.length} customer records loaded`}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleClear}
                  disabled={isPredicting}
                  className="px-3 py-2 rounded-lg border border-gray-300 bg-white text-gray-600 hover:text-red-600 hover:bg-red-50 text-xs font-medium transition-colors flex items-center gap-1 cursor-pointer disabled:opacity-50"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Remove
                </button>
                <button
                  type="button"
                  onClick={runBatchPredictions}
                  disabled={isPredicting || items.length === 0}
                  className="px-5 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold shadow-sm transition-colors flex items-center gap-2 cursor-pointer disabled:opacity-50"
                >
                  {isPredicting ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Predicting... ({progress.current}/{progress.total})
                    </>
                  ) : hasPredicted ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5" />
                      Re-run Predictions
                    </>
                  ) : (
                    <>
                      <Check className="w-3.5 h-3.5" />
                      Run Predictions ({items.length} records)
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Progress Bar */}
        {isPredicting && (
          <div className="mt-4 space-y-1.5 animate-in fade-in">
            <div className="flex justify-between text-xs text-gray-600 font-medium">
              <span>Calculating churn risk models...</span>
              <span>
                {Math.round((progress.current / (progress.total || 1)) * 100)}% ({progress.current}/{progress.total})
              </span>
            </div>
            <div className="w-full h-2 rounded-full bg-gray-100 overflow-hidden">
              <div
                className="h-full bg-blue-600 transition-all duration-200"
                style={{
                  width: `${(progress.current / (progress.total || 1)) * 100}%`,
                }}
              />
            </div>
          </div>
        )}

        {/* Error notification */}
        {error && (
          <div className="mt-4 p-4 rounded-lg bg-red-50 border border-red-200 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-red-700">{error}</div>
          </div>
        )}
      </div>

      {/* Analytics KPI Cards (after predictions) */}
      {stats && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
          <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                Total Customers
              </span>
              <Users className="w-4 h-4 text-gray-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-3xl font-extrabold text-gray-900">{stats.total}</span>
              <span className="text-xs text-gray-500">records</span>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                Predicted Churn Rate
              </span>
              <AlertTriangle className="w-4 h-4 text-red-500" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-3xl font-extrabold text-red-600">
                {stats.churnRate.toFixed(1)}%
              </span>
              <span className="text-xs text-gray-500">({stats.churnCount} customers)</span>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                High Risk Segment
              </span>
              <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-3xl font-extrabold text-gray-900">{stats.highRisk}</span>
              <span className="text-xs text-red-600 font-medium">
                ({((stats.highRisk / stats.total) * 100).toFixed(0)}%)
              </span>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                Low / Safe Segment
              </span>
              <span className="w-2.5 h-2.5 rounded-full bg-green-500" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-3xl font-extrabold text-gray-900">{stats.lowRisk}</span>
              <span className="text-xs text-green-600 font-medium">
                ({((stats.lowRisk / stats.total) * 100).toFixed(0)}%)
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Results Table Section */}
      {items.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden animate-in fade-in">
          {/* Table Toolbar */}
          <div className="p-4 sm:p-5 border-b border-gray-100 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-gray-50/50">
            <div className="flex items-center gap-2 flex-1 max-w-md">
              <div className="relative w-full">
                <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  placeholder="Search by ID, Category, Device..."
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    setCurrentPage(1);
                  }}
                  className="w-full pl-9 pr-3 py-1.5 text-xs bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600 focus:border-transparent"
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              {/* Filter Pills */}
              <div className="flex items-center gap-1 bg-gray-200/70 p-0.5 rounded-lg text-xs font-medium text-gray-700">
                {[
                  { key: "ALL", label: "All" },
                  { key: "High", label: "High Risk" },
                  { key: "Medium", label: "Med Risk" },
                  { key: "Low", label: "Low Risk" },
                  { key: "CHURN", label: "Will Churn" },
                ].map((f) => (
                  <button
                    key={f.key}
                    type="button"
                    onClick={() => {
                      setRiskFilter(f.key);
                      setCurrentPage(1);
                    }}
                    className={`px-2.5 py-1 rounded-md transition-all cursor-pointer ${
                      riskFilter === f.key
                        ? "bg-white text-gray-900 shadow-xs font-semibold"
                        : "text-gray-600 hover:text-gray-900"
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>

              {hasPredicted && (
                <button
                  type="button"
                  onClick={handleExport}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold shadow-xs transition-colors cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  Export Predictions (CSV)
                </button>
              )}
            </div>
          </div>

          {/* Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-gray-600">
              <thead className="bg-gray-100/75 text-gray-700 uppercase font-semibold border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3">Customer ID</th>
                  <th className="px-4 py-3">Prediction</th>
                  <th className="px-4 py-3">Churn Risk</th>
                  <th className="px-4 py-3">Probability</th>
                  <th className="px-4 py-3">Tenure</th>
                  <th className="px-4 py-3">Device / Order Cat</th>
                  <th className="px-4 py-3">Payment</th>
                  <th className="px-4 py-3">Complaint?</th>
                  <th className="px-4 py-3">Cashback</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {paginatedItems.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="text-center py-10 text-gray-400">
                      No matching records found.
                    </td>
                  </tr>
                ) : (
                  paginatedItems.map((item, idx) => {
                    const pred = item.prediction;
                    return (
                      <tr key={idx} className="hover:bg-gray-50/80 transition-colors">
                        <td className="px-4 py-3 font-semibold text-gray-900">
                          {item.id}
                        </td>
                        <td className="px-4 py-3">
                          {pred ? (
                            <span
                              className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${
                                pred.churn_prediction
                                  ? "bg-red-100 text-red-700"
                                  : "bg-green-100 text-green-700"
                              }`}
                            >
                              {pred.churn_prediction ? "Will Churn" : "Will Stay"}
                            </span>
                          ) : (
                            <span className="text-gray-400 italic">Pending</span>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          {pred ? (
                            <span
                              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium ${
                                pred.risk_tier === "High"
                                  ? "bg-red-50 text-red-700 border border-red-200"
                                  : pred.risk_tier === "Medium"
                                  ? "bg-yellow-50 text-yellow-700 border border-yellow-200"
                                  : "bg-green-50 text-green-700 border border-green-200"
                              }`}
                            >
                              <span
                                className={`w-1.5 h-1.5 rounded-full ${
                                  pred.risk_tier === "High"
                                    ? "bg-red-500"
                                    : pred.risk_tier === "Medium"
                                    ? "bg-yellow-500"
                                    : "bg-green-500"
                                }`}
                              />
                              {pred.risk_tier} Risk
                            </span>
                          ) : (
                            "—"
                          )}
                        </td>
                        <td className="px-4 py-3 font-medium text-gray-900">
                          {pred ? (
                            <div className="flex items-center gap-2">
                              <span>{(pred.churn_probability * 100).toFixed(1)}%</span>
                              <div className="w-12 h-1.5 rounded-full bg-gray-100 overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${
                                    pred.churn_probability > 0.6
                                      ? "bg-red-500"
                                      : pred.churn_probability > 0.3
                                      ? "bg-yellow-500"
                                      : "bg-green-500"
                                  }`}
                                  style={{ width: `${pred.churn_probability * 100}%` }}
                                />
                              </div>
                            </div>
                          ) : (
                            "—"
                          )}
                        </td>
                        <td className="px-4 py-3">{item.customer.Tenure} mo</td>
                        <td className="px-4 py-3">
                          <div>
                            <span className="font-medium text-gray-800">
                              {item.customer.PreferredLoginDevice}
                            </span>
                            <span className="text-gray-400 block text-[11px]">
                              {item.customer.PreferedOrderCat}
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-3">{item.customer.PreferredPaymentMode}</td>
                        <td className="px-4 py-3">
                          {item.customer.Complain === 1 ? (
                            <span className="text-red-600 font-semibold">Yes</span>
                          ) : (
                            <span className="text-gray-400">No</span>
                          )}
                        </td>
                        <td className="px-4 py-3 font-medium">${item.customer.CashbackAmount}</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          {/* Table Pagination Footer */}
          <div className="p-4 border-t border-gray-100 flex items-center justify-between text-xs text-gray-500 bg-gray-50/30">
            <div>
              Showing{" "}
              <span className="font-semibold text-gray-800">
                {filteredItems.length > 0 ? (currentPage - 1) * pageSize + 1 : 0}
              </span>{" "}
              to{" "}
              <span className="font-semibold text-gray-800">
                {Math.min(currentPage * pageSize, filteredItems.length)}
              </span>{" "}
              of <span className="font-semibold text-gray-800">{filteredItems.length}</span>{" "}
              records
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="p-1.5 rounded-md border border-gray-200 bg-white hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span>
                Page {currentPage} of {totalPages}
              </span>
              <button
                type="button"
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="p-1.5 rounded-md border border-gray-200 bg-white hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
