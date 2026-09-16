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

  const inputClasses = "w-full border border-black p-2 mt-1 mb-5 bg-white text-black focus:outline-none focus:border-[#1e3a8a] focus:ring-1 focus:ring-[#1e3a8a]";
  const labelClasses = "block font-bold text-sm text-black tracking-wide";
  const sectionTitleClasses = "text-xl font-bold uppercase border-b border-black pb-2 mb-6 mt-8";

  return (
    <div className="min-h-screen bg-white text-black p-4 md:p-8 font-sans">
      <div className="max-w-4xl mx-auto border-2 border-black p-6 md:p-10 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
        <header className="border-b-2 border-black pb-6 mb-2 flex flex-col md:flex-row md:justify-between md:items-end gap-4">
          <div>
            <h1 className="text-4xl font-black uppercase tracking-tight mb-2">Churn Prediction</h1>
            <p className="text-gray-600 font-mono text-sm max-w-md">
              Enter customer data below to predict their likelihood of churning.
            </p>
          </div>
          <div className="flex flex-col gap-2 text-xs font-mono text-right">
            <div className="border border-black px-3 py-1 inline-block self-start md:self-end bg-gray-50">
              <span className="font-bold">STATUS:</span> {status}
            </div>
            {metrics && (
              <div className="border border-black px-3 py-1 inline-block bg-gray-50">
                <span className="font-bold">MODEL:</span> {metrics.model_name} | <span className="font-bold">ACCURACY:</span> {(metrics.accuracy * 100).toFixed(1)}%
              </div>
            )}
          </div>
        </header>

        <form onSubmit={handleSubmit}>
          
          <h2 className={sectionTitleClasses}>1. Demographics & Location</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
            <div>
              <label className={labelClasses}>Gender</label>
              <select name="Gender" required className={inputClasses}>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>

              <label className={labelClasses}>Marital Status</label>
              <select name="MaritalStatus" required className={inputClasses}>
                <option value="Single">Single</option>
                <option value="Married">Married</option>
                <option value="Divorced">Divorced</option>
              </select>
            </div>
            <div>
              <label className={labelClasses}>City Tier</label>
              <select name="CityTier" required className={inputClasses}>
                <option value="1">Tier 1</option>
                <option value="2">Tier 2</option>
                <option value="3">Tier 3</option>
              </select>

              <label className={labelClasses}>Number Of Addresses</label>
              <input type="number" name="NumberOfAddress" required min="1" className={inputClasses} defaultValue="1" />
            </div>
          </div>

          <h2 className={sectionTitleClasses}>2. App Usage & Preferences</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
            <div>
              <label className={labelClasses}>Preferred Login Device</label>
              <select name="PreferredLoginDevice" required className={inputClasses}>
                <option value="Mobile Phone">Mobile Phone</option>
                <option value="Computer">Computer</option>
                <option value="Phone">Phone</option>
              </select>

              <label className={labelClasses}>Hours Spent On App</label>
              <input type="number" step="any" min="0" name="HourSpendOnApp" required className={inputClasses} defaultValue="2" />

              <label className={labelClasses}>Number Of Devices Registered</label>
              <input type="number" name="NumberOfDeviceRegistered" required min="1" className={inputClasses} defaultValue="1" />
            </div>
            <div>
              <label className={labelClasses}>Preferred Payment Mode</label>
              <select name="PreferredPaymentMode" required className={inputClasses}>
                <option value="Debit Card">Debit Card</option>
                <option value="Credit Card">Credit Card</option>
                <option value="E wallet">E-Wallet</option>
                <option value="UPI">UPI</option>
                <option value="Cash on Delivery">Cash on Delivery</option>
              </select>

              <label className={labelClasses}>Preferred Order Category</label>
              <select name="PreferedOrderCat" required className={inputClasses}>
                <option value="Laptop & Accessory">Laptop & Accessory</option>
                <option value="Mobile">Mobile</option>
                <option value="Fashion">Fashion</option>
                <option value="Grocery">Grocery</option>
                <option value="Others">Others</option>
              </select>
            </div>
          </div>

          <h2 className={sectionTitleClasses}>3. Order History & Engagement</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
            <div>
              <label className={labelClasses}>Tenure (Months)</label>
              <input type="number" step="any" min="0" name="Tenure" required className={inputClasses} defaultValue="0" />

              <label className={labelClasses}>Satisfaction Score (1-5)</label>
              <input type="number" name="SatisfactionScore" required min="1" max="5" className={inputClasses} defaultValue="3" />

              <label className={labelClasses}>Complained in last month?</label>
              <select name="Complain" required className={inputClasses}>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>

              <label className={labelClasses}>Warehouse To Home (km)</label>
              <input type="number" step="any" min="0" name="WarehouseToHome" required className={inputClasses} defaultValue="10" />
            </div>
            <div>
              <label className={labelClasses}>Order Amount Hike From Last Year (%)</label>
              <input type="number" step="any" min="0" name="OrderAmountHikeFromlastYear" required className={inputClasses} defaultValue="10" />

              <label className={labelClasses}>Coupons Used</label>
              <input type="number" step="any" min="0" name="CouponUsed" required className={inputClasses} defaultValue="0" />

              <label className={labelClasses}>Total Order Count</label>
              <input type="number" step="any" min="0" name="OrderCount" required className={inputClasses} defaultValue="1" />

              <label className={labelClasses}>Days Since Last Order</label>
              <input type="number" step="any" min="0" name="DaySinceLastOrder" required className={inputClasses} defaultValue="0" />

              <label className={labelClasses}>Cashback Amount ($)</label>
              <input type="number" step="any" min="0" name="CashbackAmount" required className={inputClasses} defaultValue="0" />
            </div>
          </div>

          <div className="mt-10">
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-black text-white font-black text-lg uppercase tracking-widest py-5 border-2 border-black hover:bg-[#1e3a8a] hover:border-[#1e3a8a] hover:shadow-[4px_4px_0px_0px_rgba(30,58,138,1)] transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none"
            >
              {loading ? "Running Prediction..." : "Run Prediction"}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-8 border-2 border-black bg-gray-50 p-6 text-center font-bold font-mono">
            ⚠️ ERROR: {error}
          </div>
        )}

        {prediction && (
          <div className="mt-10 border-4 border-black p-8 text-center bg-gray-50 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
            <h2 className="text-3xl font-black uppercase mb-6 border-b-4 border-black pb-4 inline-block">Prediction Result</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 font-mono text-xl mt-2">
              <div className="border-2 border-black p-6 bg-white">
                <div className="font-bold text-sm uppercase text-gray-500 mb-2">Probability</div>
                <div className="text-4xl font-black">{(prediction.churn_probability * 100).toFixed(1)}%</div>
              </div>
              <div className="border-2 border-black p-6 bg-white">
                <div className="font-bold text-sm uppercase text-gray-500 mb-2">Outcome</div>
                <div className={`text-3xl font-black mt-2 ${prediction.churn_prediction ? 'text-black' : 'text-gray-600'}`}>
                  {prediction.churn_prediction ? "WILL CHURN" : "WILL STAY"}
                </div>
              </div>
              <div className="border-2 border-black p-6 bg-white">
                <div className="font-bold text-sm uppercase text-gray-500 mb-2">Risk Tier</div>
                <div className="text-3xl font-black mt-2">{prediction.risk_tier}</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
