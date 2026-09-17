"use client";

import { useState, useEffect } from "react";

export default function Home() {
  const [status, setStatus] = useState("Checking backend...");
  const [metrics, setMetrics] = useState<any>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const rawUrl = process.env.NEXT_PUBLIC_API_URL || "https://web-production-98eb2.up.railway.app";
  const apiUrl = rawUrl.includes("84f12")
    ? "https://web-production-98eb2.up.railway.app"
    : rawUrl.replace(/\/$/, "");

  useEffect(() => {
    fetch(`${apiUrl}/health`)
      .then((res) => {
        if (res.ok) {
          setStatus("Online");
        } else {
          setStatus("Error");
        }
      })
      .catch((err) => {
        console.error("Health check error:", err);
        setStatus("Offline");
      });

    fetch(`${apiUrl}/model/metadata`)
      .then((res) => res.json())
      .then((data) => setMetrics(data))
      .catch(() => {});
  }, [apiUrl]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setPrediction(null);

    const formData = new FormData(e.currentTarget);
    const payload: Record<string, any> = {};

    formData.forEach((value, key) => {
      const numericFields = [
        "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
        "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
        "Complain", "OrderAmountHikeFromlastYear", "CouponUsed",
        "OrderCount", "DaySinceLastOrder", "CashbackAmount"
      ];
      if (numericFields.includes(key)) {
        payload[key] = Number(value);
      } else {
        payload[key] = value;
      }
    });

    try {
      const res = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error("Failed to predict. Check your inputs.");
      }

      const data = await res.json();
      setPrediction(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const inputStyles = "flex h-10 w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-600 focus:border-transparent disabled:cursor-not-allowed disabled:opacity-50 transition-all";
  const labelStyles = "text-sm font-semibold leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 text-gray-700 mb-1.5 block";
  const cardStyles = "bg-white border border-gray-200 rounded-xl shadow-sm p-6";

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 py-12 px-4 sm:px-6 lg:px-8 font-sans">
      <div className="max-w-5xl mx-auto space-y-8">
        
        {/* Header Section */}
        <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight text-gray-900 sm:text-4xl">Customer Churn Prediction</h1>
            <p className="mt-2 text-sm text-gray-500 max-w-xl">
              Fill out the customer profile below to predict their likelihood of churning. Our machine learning model analyzes behavioral and demographic data in real-time.
            </p>
          </div>
          <div className="flex flex-col gap-2">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-gray-200 shadow-sm text-sm font-medium text-gray-700 w-fit">
              <span className={`w-2.5 h-2.5 rounded-full ${status === 'Online' ? 'bg-green-500' : 'bg-red-500 animate-pulse'}`}></span>
              API: {status}
            </div>
            {metrics && (
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-gray-200 shadow-sm text-xs font-medium text-gray-600 w-fit">
                <svg className="w-4 h-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                </svg>
                {metrics.model_name} • {(metrics.accuracy * 100).toFixed(1)}% Acc
              </div>
            )}
          </div>
        </header>

        <form onSubmit={handleSubmit} className="space-y-6">
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Column 1: Demographics */}
            <div className={cardStyles}>
              <div className="mb-5 border-b border-gray-100 pb-4">
                <h2 className="text-lg font-semibold text-gray-900">Demographics</h2>
                <p className="text-xs text-gray-500 mt-1">Personal and location details.</p>
              </div>
              <div className="space-y-4">
                <div>
                  <label className={labelStyles}>Gender</label>
                  <select name="Gender" required className={inputStyles}>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>
                <div>
                  <label className={labelStyles}>Marital Status</label>
                  <select name="MaritalStatus" required className={inputStyles}>
                    <option value="Single">Single</option>
                    <option value="Married">Married</option>
                    <option value="Divorced">Divorced</option>
                  </select>
                </div>
                <div>
                  <label className={labelStyles}>City Tier</label>
                  <select name="CityTier" required className={inputStyles}>
                    <option value="1">Tier 1 (Metro)</option>
                    <option value="2">Tier 2</option>
                    <option value="3">Tier 3</option>
                  </select>
                </div>
                <div>
                  <label className={labelStyles}>Number Of Addresses</label>
                  <input type="number" name="NumberOfAddress" required min="1" className={inputStyles} defaultValue="1" />
                </div>
                <div>
                  <label className={labelStyles}>Warehouse To Home (km)</label>
                  <input type="number" step="any" min="0" name="WarehouseToHome" required className={inputStyles} defaultValue="10" />
                </div>
              </div>
            </div>

            {/* Column 2: App Usage & Preferences */}
            <div className={cardStyles}>
              <div className="mb-5 border-b border-gray-100 pb-4">
                <h2 className="text-lg font-semibold text-gray-900">Preferences</h2>
                <p className="text-xs text-gray-500 mt-1">How the customer interacts with us.</p>
              </div>
              <div className="space-y-4">
                <div>
                  <label className={labelStyles}>Preferred Login Device</label>
                  <select name="PreferredLoginDevice" required className={inputStyles}>
                    <option value="Mobile Phone">Mobile Phone</option>
                    <option value="Computer">Computer</option>
                  </select>
                </div>
                <div>
                  <label className={labelStyles}>Preferred Payment Mode</label>
                  <select name="PreferredPaymentMode" required className={inputStyles}>
                    <option value="Debit Card">Debit Card</option>
                    <option value="Credit Card">Credit Card</option>
                    <option value="E wallet">E-Wallet</option>
                    <option value="UPI">UPI</option>
                    <option value="Cash on Delivery">Cash on Delivery</option>
                  </select>
                </div>
                <div>
                  <label className={labelStyles}>Preferred Order Category</label>
                  <select name="PreferedOrderCat" required className={inputStyles}>
                    <option value="Laptop & Accessory">Laptop & Accessory</option>
                    <option value="Mobile">Mobile</option>
                    <option value="Fashion">Fashion</option>
                    <option value="Grocery">Grocery</option>
                    <option value="Others">Others</option>
                  </select>
                </div>
                <div>
                  <label className={labelStyles}>Hours Spent On App (Daily)</label>
                  <input type="number" step="any" min="0" name="HourSpendOnApp" required className={inputStyles} defaultValue="2" />
                </div>
                <div>
                  <label className={labelStyles}>Devices Registered</label>
                  <input type="number" name="NumberOfDeviceRegistered" required min="1" className={inputStyles} defaultValue="1" />
                </div>
              </div>
            </div>

            {/* Column 3: Order History & Engagement */}
            <div className={cardStyles}>
              <div className="mb-5 border-b border-gray-100 pb-4">
                <h2 className="text-lg font-semibold text-gray-900">Engagement</h2>
                <p className="text-xs text-gray-500 mt-1">Purchase history and satisfaction.</p>
              </div>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className={labelStyles}>Tenure (Months)</label>
                    <input type="number" step="any" min="0" name="Tenure" required className={inputStyles} defaultValue="12" />
                  </div>
                  <div>
                    <label className={labelStyles}>Satisfaction (1-5)</label>
                    <input type="number" name="SatisfactionScore" required min="1" max="5" className={inputStyles} defaultValue="3" />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className={labelStyles}>Order Count</label>
                    <input type="number" step="any" min="0" name="OrderCount" required className={inputStyles} defaultValue="5" />
                  </div>
                  <div>
                    <label className={labelStyles}>Days Since Order</label>
                    <input type="number" step="any" min="0" name="DaySinceLastOrder" required className={inputStyles} defaultValue="7" />
                  </div>
                </div>
                <div>
                  <label className={labelStyles}>Order Amount Hike (%)</label>
                  <input type="number" step="any" min="0" name="OrderAmountHikeFromlastYear" required className={inputStyles} defaultValue="10" />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className={labelStyles}>Coupons Used</label>
                    <input type="number" step="any" min="0" name="CouponUsed" required className={inputStyles} defaultValue="1" />
                  </div>
                  <div>
                    <label className={labelStyles}>Cashback ($)</label>
                    <input type="number" step="any" min="0" name="CashbackAmount" required className={inputStyles} defaultValue="50" />
                  </div>
                </div>
                <div>
                  <label className={labelStyles}>Recent Complaint?</label>
                  <select name="Complain" required className={inputStyles}>
                    <option value="0">No Complaints</option>
                    <option value="1">Yes, Complained</option>
                  </select>
                </div>
              </div>
            </div>
            
          </div>

          <div className="flex justify-end pt-4">
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center justify-center rounded-lg bg-blue-600 px-8 py-3.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </>
              ) : (
                "Predict Customer Churn"
              )}
            </button>
          </div>
        </form>

        {error && (
          <div className="rounded-lg bg-red-50 p-4 border border-red-200 mt-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Prediction Error</h3>
                <div className="mt-2 text-sm text-red-700">{error}</div>
              </div>
            </div>
          </div>
        )}

        {prediction && (
          <div className="mt-8 rounded-2xl bg-white border border-gray-200 shadow-lg overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="px-6 py-5 border-b border-gray-100 bg-gray-50/50 flex justify-between items-center">
              <h3 className="text-lg font-semibold text-gray-900">Analysis Results</h3>
              <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${prediction.churn_prediction ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                {prediction.churn_prediction ? 'High Risk' : 'Low Risk'}
              </span>
            </div>
            <div className="p-6">
              <dl className="grid grid-cols-1 gap-x-4 gap-y-8 sm:grid-cols-3">
                <div className="sm:col-span-1">
                  <dt className="text-sm font-medium text-gray-500">Churn Probability</dt>
                  <dd className="mt-2 text-4xl font-extrabold tracking-tight text-gray-900">
                    {(prediction.churn_probability * 100).toFixed(1)}<span className="text-2xl text-gray-400">%</span>
                  </dd>
                </div>
                <div className="sm:col-span-1">
                  <dt className="text-sm font-medium text-gray-500">Predicted Outcome</dt>
                  <dd className={`mt-2 text-2xl font-bold ${prediction.churn_prediction ? 'text-red-600' : 'text-green-600'}`}>
                    {prediction.churn_prediction ? "Will Churn" : "Will Stay"}
                  </dd>
                </div>
                <div className="sm:col-span-1">
                  <dt className="text-sm font-medium text-gray-500">Risk Tier Assessment</dt>
                  <dd className="mt-2">
                    <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-semibold ${
                      prediction.risk_tier === 'High' ? 'bg-red-50 text-red-700 border border-red-200' : 
                      prediction.risk_tier === 'Medium' ? 'bg-yellow-50 text-yellow-700 border border-yellow-200' : 
                      'bg-green-50 text-green-700 border border-green-200'
                    }`}>
                      <span className={`w-2 h-2 rounded-full ${
                        prediction.risk_tier === 'High' ? 'bg-red-500' : 
                        prediction.risk_tier === 'Medium' ? 'bg-yellow-500' : 
                        'bg-green-500'
                      }`}></span>
                      {prediction.risk_tier} Tier
                    </span>
                  </dd>
                </div>
              </dl>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
