"use client";

import { useState, useEffect } from "react";

export default function Home() {
  const [status, setStatus] = useState<string>("Checking backend...");
  const [metrics, setMetrics] = useState<any>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  useEffect(() => {
    fetch(`${apiUrl}/health`)
      .then((res) => (res.ok ? setStatus("Backend Online") : setStatus("Backend Error")))
      .catch(() => setStatus("Backend Offline"));

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
      // Convert numeric fields
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

  const inputClasses = "w-full border border-black p-2 mt-1 mb-4 bg-white text-black focus:outline-none focus:border-[#1e3a8a] focus:ring-1 focus:ring-[#1e3a8a]";
  const labelClasses = "block font-bold text-black";

  return (
    <div className="min-h-screen bg-white text-black p-8 font-sans">
      <div className="max-w-4xl mx-auto border-2 border-black p-6">
        <header className="border-b-2 border-black pb-4 mb-6">
          <h1 className="text-3xl font-bold uppercase tracking-wider mb-2">Churn Prediction</h1>
          <div className="flex flex-wrap items-center gap-4 text-sm font-mono">
            <div className="border border-black px-2 py-1">Status: {status}</div>
            {metrics && (
              <>
                <div className="border border-black px-2 py-1">Model: {metrics.model_name}</div>
                <div className="border border-black px-2 py-1">Accuracy: {(metrics.accuracy * 100).toFixed(2)}%</div>
                <div className="border border-black px-2 py-1">F1: {metrics.f1.toFixed(4)}</div>
              </>
            )}
          </div>
        </header>

        <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
          <div>
            <label className={labelClasses}>Tenure (Months)</label>
            <input type="number" step="any" name="Tenure" required className={inputClasses} defaultValue="0" />
            
            <label className={labelClasses}>Preferred Login Device</label>
            <select name="PreferredLoginDevice" required className={inputClasses}>
              <option value="Mobile Phone">Mobile Phone</option>
              <option value="Computer">Computer</option>
              <option value="Phone">Phone</option>
            </select>

            <label className={labelClasses}>City Tier (1, 2, or 3)</label>
            <input type="number" name="CityTier" required min="1" max="3" className={inputClasses} defaultValue="1" />

            <label className={labelClasses}>Warehouse To Home (km)</label>
            <input type="number" step="any" name="WarehouseToHome" required className={inputClasses} defaultValue="0" />

            <label className={labelClasses}>Preferred Payment Mode</label>
            <select name="PreferredPaymentMode" required className={inputClasses}>
              <option value="Debit Card">Debit Card</option>
              <option value="Credit Card">Credit Card</option>
              <option value="E wallet">E wallet</option>
              <option value="UPI">UPI</option>
              <option value="Cash on Delivery">Cash on Delivery</option>
              <option value="CC">CC</option>
              <option value="COD">COD</option>
            </select>

            <label className={labelClasses}>Gender</label>
            <select name="Gender" required className={inputClasses}>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>

            <label className={labelClasses}>Hours Spent On App</label>
            <input type="number" step="any" name="HourSpendOnApp" required className={inputClasses} defaultValue="0" />

            <label className={labelClasses}>Number Of Devices Registered</label>
            <input type="number" name="NumberOfDeviceRegistered" required className={inputClasses} defaultValue="1" />

            <label className={labelClasses}>Preferred Order Category</label>
            <select name="PreferedOrderCat" required className={inputClasses}>
              <option value="Laptop & Accessory">Laptop & Accessory</option>
              <option value="Mobile">Mobile</option>
              <option value="Mobile Phone">Mobile Phone</option>
              <option value="Others">Others</option>
              <option value="Fashion">Fashion</option>
              <option value="Grocery">Grocery</option>
            </select>
          </div>

          <div>
            <label className={labelClasses}>Satisfaction Score (1-5)</label>
            <input type="number" name="SatisfactionScore" required min="1" max="5" className={inputClasses} defaultValue="3" />

            <label className={labelClasses}>Marital Status</label>
            <select name="MaritalStatus" required className={inputClasses}>
              <option value="Single">Single</option>
              <option value="Married">Married</option>
              <option value="Divorced">Divorced</option>
            </select>

            <label className={labelClasses}>Number Of Addresses</label>
            <input type="number" name="NumberOfAddress" required className={inputClasses} defaultValue="1" />

            <label className={labelClasses}>Complain in last month (0 or 1)</label>
            <input type="number" name="Complain" required min="0" max="1" className={inputClasses} defaultValue="0" />

            <label className={labelClasses}>Order Amount Hike From Last Year (%)</label>
            <input type="number" step="any" name="OrderAmountHikeFromlastYear" required className={inputClasses} defaultValue="0" />

            <label className={labelClasses}>Coupons Used</label>
            <input type="number" step="any" name="CouponUsed" required className={inputClasses} defaultValue="0" />

            <label className={labelClasses}>Total Order Count</label>
            <input type="number" step="any" name="OrderCount" required className={inputClasses} defaultValue="0" />

            <label className={labelClasses}>Days Since Last Order</label>
            <input type="number" step="any" name="DaySinceLastOrder" required className={inputClasses} defaultValue="0" />

            <label className={labelClasses}>Cashback Amount</label>
            <input type="number" step="any" name="CashbackAmount" required className={inputClasses} defaultValue="0" />
          </div>

          <div className="md:col-span-2 mt-6">
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-black text-white font-bold uppercase tracking-widest py-4 border-2 border-black hover:bg-[#1e3a8a] hover:border-[#1e3a8a] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Predicting..." : "Run Prediction"}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-6 border-2 border-black p-4 text-center font-bold">
            Error: {error}
          </div>
        )}

        {prediction && (
          <div className="mt-8 border-4 border-black p-6 text-center">
            <h2 className="text-2xl font-bold uppercase mb-4 border-b-2 border-black pb-2 inline-block">Prediction Result</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-lg mt-4">
              <div className="border border-black p-4">
                <div className="font-bold text-sm uppercase text-gray-500 mb-1">Probability</div>
                <div className="text-3xl">{(prediction.churn_probability * 100).toFixed(1)}%</div>
              </div>
              <div className="border border-black p-4">
                <div className="font-bold text-sm uppercase text-gray-500 mb-1">Outcome</div>
                <div className="text-2xl mt-1">{prediction.churn_prediction ? "WILL CHURN" : "WILL STAY"}</div>
              </div>
              <div className="border border-black p-4">
                <div className="font-bold text-sm uppercase text-gray-500 mb-1">Risk Tier</div>
                <div className="text-2xl mt-1">{prediction.risk_tier}</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
