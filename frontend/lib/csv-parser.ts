/**
 * Robust CSV parser and serializer compliant with RFC 4180.
 * Zero external dependencies.
 */

export interface ParsedCSV {
  headers: string[];
  rows: Record<string, string>[];
}

export interface CustomerInput {
  Tenure: number;
  PreferredLoginDevice: string;
  CityTier: number;
  WarehouseToHome: number;
  PreferredPaymentMode: string;
  Gender: string;
  HourSpendOnApp: number;
  NumberOfDeviceRegistered: number;
  PreferedOrderCat: string;
  SatisfactionScore: number;
  MaritalStatus: string;
  NumberOfAddress: number;
  Complain: number;
  OrderAmountHikeFromlastYear: number;
  CouponUsed: number;
  OrderCount: number;
  DaySinceLastOrder: number;
  CashbackAmount: number;
}

export interface PredictionResult {
  churn_probability: number;
  churn_prediction: boolean;
  risk_tier: "Low" | "Medium" | "High";
}

export interface BatchItem {
  id: string | number;
  raw: Record<string, string>;
  customer: CustomerInput;
  prediction?: PredictionResult;
  error?: string;
}

/**
 * Parse CSV text into headers and an array of row objects
 */
export function parseCSV(csvText: string): ParsedCSV {
  const lines: string[][] = [];
  let currentLine: string[] = [];
  let currentToken = "";
  let inQuotes = false;

  const text = csvText.replace(/\r\n/g, "\n").replace(/\r/g, "\n");

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const nextChar = text[i + 1];

    if (inQuotes) {
      if (char === '"' && nextChar === '"') {
        currentToken += '"';
        i++; // skip next quote
      } else if (char === '"') {
        inQuotes = false;
      } else {
        currentToken += char;
      }
    } else {
      if (char === '"') {
        inQuotes = true;
      } else if (char === ",") {
        currentLine.push(currentToken.trim());
        currentToken = "";
      } else if (char === "\n") {
        currentLine.push(currentToken.trim());
        if (currentLine.some((col) => col.length > 0)) {
          lines.push(currentLine);
        }
        currentLine = [];
        currentToken = "";
      } else {
        currentToken += char;
      }
    }
  }

  // Last token / line if present
  if (currentToken.length > 0 || currentLine.length > 0) {
    currentLine.push(currentToken.trim());
    if (currentLine.some((col) => col.length > 0)) {
      lines.push(currentLine);
    }
  }

  if (lines.length === 0) {
    return { headers: [], rows: [] };
  }

  const rawHeaders = lines[0];
  const headers = rawHeaders.map((h) => h.trim().replace(/^[\uFEFF]/, "")); // remove BOM if present

  const rows: Record<string, string>[] = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i];
    const rowObj: Record<string, string> = {};
    headers.forEach((header, index) => {
      rowObj[header] = values[index] !== undefined ? values[index] : "";
    });
    rows.push(rowObj);
  }

  return { headers, rows };
}

/**
 * Column alias mapping to canonical Customer fields
 */
const FIELD_ALIASES: Record<string, keyof CustomerInput> = {
  tenure: "Tenure",
  preferredlogindevice: "PreferredLoginDevice",
  logindevice: "PreferredLoginDevice",
  device: "PreferredLoginDevice",
  citytier: "CityTier",
  tier: "CityTier",
  warehousetohome: "WarehouseToHome",
  warehousedistance: "WarehouseToHome",
  distance: "WarehouseToHome",
  preferredpaymentmode: "PreferredPaymentMode",
  paymentmode: "PreferredPaymentMode",
  paymentmethod: "PreferredPaymentMode",
  gender: "Gender",
  sex: "Gender",
  hourspendonapp: "HourSpendOnApp",
  hoursspendonapp: "HourSpendOnApp",
  hoursspent: "HourSpendOnApp",
  numberofdeviceregistered: "NumberOfDeviceRegistered",
  deviceregistered: "NumberOfDeviceRegistered",
  devices: "NumberOfDeviceRegistered",
  preferedordercat: "PreferedOrderCat",
  preferredordercat: "PreferedOrderCat",
  ordercat: "PreferedOrderCat",
  category: "PreferedOrderCat",
  satisfactionscore: "SatisfactionScore",
  satisfaction: "SatisfactionScore",
  maritalstatus: "MaritalStatus",
  marital: "MaritalStatus",
  numberofaddress: "NumberOfAddress",
  addresses: "NumberOfAddress",
  addresscount: "NumberOfAddress",
  complain: "Complain",
  complaint: "Complain",
  complained: "Complain",
  orderamounthikefromlastyear: "OrderAmountHikeFromlastYear",
  orderamounthike: "OrderAmountHikeFromlastYear",
  hike: "OrderAmountHikeFromlastYear",
  couponused: "CouponUsed",
  coupon: "CouponUsed",
  coupons: "CouponUsed",
  ordercount: "OrderCount",
  orders: "OrderCount",
  daysincelastorder: "DaySinceLastOrder",
  dayssinceorder: "DaySinceLastOrder",
  cashbackamount: "CashbackAmount",
  cashback: "CashbackAmount",
};

/**
 * Standard Defaults for any missing values
 */
const DEFAULTS: CustomerInput = {
  Tenure: 12,
  PreferredLoginDevice: "Mobile Phone",
  CityTier: 1,
  WarehouseToHome: 10,
  PreferredPaymentMode: "Debit Card",
  Gender: "Male",
  HourSpendOnApp: 2,
  NumberOfDeviceRegistered: 1,
  PreferedOrderCat: "Mobile",
  SatisfactionScore: 3,
  MaritalStatus: "Single",
  NumberOfAddress: 1,
  Complain: 0,
  OrderAmountHikeFromlastYear: 10,
  CouponUsed: 1,
  OrderCount: 1,
  DaySinceLastOrder: 5,
  CashbackAmount: 50,
};

/**
 * Map arbitrary CSV row object to strict CustomerInput
 */
export function mapRowToCustomer(row: Record<string, string>, index: number): BatchItem {
  const customer: CustomerInput = { ...DEFAULTS };
  let detectedId: string | number = `Customer #${index + 1}`;

  for (const [key, rawVal] of Object.entries(row)) {
    const cleanKey = key.trim().toLowerCase().replace(/[^a-z0-9]/g, "");
    const val = rawVal ? rawVal.trim() : "";

    if (cleanKey === "customerid" || cleanKey === "id" || cleanKey === "clientid") {
      if (val) detectedId = val;
      continue;
    }

    const matchedField = FIELD_ALIASES[cleanKey];
    if (matchedField) {
      if (typeof DEFAULTS[matchedField] === "number") {
        const parsedNum = parseFloat(val);
        (customer as any)[matchedField] = !isNaN(parsedNum) ? parsedNum : DEFAULTS[matchedField];
      } else {
        // String field normalizations
        if (matchedField === "PreferredLoginDevice") {
          if (/phone|mobile/i.test(val)) customer.PreferredLoginDevice = "Mobile Phone";
          else if (/comp|pc|laptop/i.test(val)) customer.PreferredLoginDevice = "Computer";
          else customer.PreferredLoginDevice = val || DEFAULTS.PreferredLoginDevice;
        } else if (matchedField === "PreferredPaymentMode") {
          if (/debit/i.test(val)) customer.PreferredPaymentMode = "Debit Card";
          else if (/credit|cc/i.test(val)) customer.PreferredPaymentMode = "Credit Card";
          else if (/wallet/i.test(val)) customer.PreferredPaymentMode = "E wallet";
          else if (/upi/i.test(val)) customer.PreferredPaymentMode = "UPI";
          else if (/cod|cash/i.test(val)) customer.PreferredPaymentMode = "Cash on Delivery";
          else customer.PreferredPaymentMode = val || DEFAULTS.PreferredPaymentMode;
        } else if (matchedField === "Gender") {
          if (/^f/i.test(val)) customer.Gender = "Female";
          else customer.Gender = "Male";
        } else if (matchedField === "MaritalStatus") {
          if (/marr/i.test(val)) customer.MaritalStatus = "Married";
          else if (/div/i.test(val)) customer.MaritalStatus = "Divorced";
          else customer.MaritalStatus = "Single";
        } else if (matchedField === "PreferedOrderCat") {
          if (/laptop|accessory/i.test(val)) customer.PreferedOrderCat = "Laptop & Accessory";
          else if (/mobile/i.test(val)) customer.PreferedOrderCat = "Mobile";
          else if (/fash/i.test(val)) customer.PreferedOrderCat = "Fashion";
          else if (/groc/i.test(val)) customer.PreferedOrderCat = "Grocery";
          else customer.PreferedOrderCat = val || DEFAULTS.PreferedOrderCat;
        } else {
          (customer as any)[matchedField] = val || DEFAULTS[matchedField];
        }
      }
    }
  }

  return {
    id: detectedId,
    raw: row,
    customer,
  };
}

/**
 * Convert data to CSV string and download in browser
 */
export function exportToCSV(
  data: Record<string, any>[],
  fileName = "churn_predictions_export.csv"
) {
  if (!data || data.length === 0) return;

  const headers = Object.keys(data[0]);

  const escapeCSV = (value: any): string => {
    if (value === null || value === undefined) return "";
    const str = String(value);
    if (str.includes(",") || str.includes('"') || str.includes("\n")) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };

  const csvRows: string[] = [];
  csvRows.push(headers.map(escapeCSV).join(","));

  for (const row of data) {
    const values = headers.map((header) => escapeCSV(row[header]));
    csvRows.push(values.join(","));
  }

  const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", fileName);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export const SAMPLE_CSV_CONTENT = `CustomerID,Tenure,PreferredLoginDevice,CityTier,WarehouseToHome,PreferredPaymentMode,Gender,HourSpendOnApp,NumberOfDeviceRegistered,PreferedOrderCat,SatisfactionScore,MaritalStatus,NumberOfAddress,Complain,OrderAmountHikeFromlastYear,CouponUsed,OrderCount,DaySinceLastOrder,CashbackAmount
50001,4.0,Mobile Phone,3,6.0,Debit Card,Female,3.0,3,Laptop & Accessory,2,Single,9,1,11.0,1.0,1.0,5.0,159.93
50002,1.0,Phone,1,8.0,UPI,Male,3.0,4,Mobile,3,Single,7,1,15.0,0.0,1.0,0.0,120.90
50003,14.0,Phone,1,30.0,Debit Card,Male,2.0,4,Mobile,3,Single,6,0,14.0,0.0,1.0,3.0,120.28
50004,0.0,Phone,3,15.0,Debit Card,Male,2.0,4,Laptop & Accessory,5,Single,8,0,23.0,0.0,1.0,3.0,134.07
50005,0.0,Phone,1,12.0,CC,Male,2.0,3,Mobile,5,Single,3,0,11.0,1.0,1.0,3.0,129.60
50006,0.0,Computer,1,22.0,Debit Card,Female,3.0,5,Mobile Phone,5,Single,2,1,22.0,4.0,6.0,7.0,139.19
50007,8.0,Phone,3,11.0,Cash on Delivery,Male,2.0,3,Laptop & Accessory,2,Divorced,4,0,14.0,0.0,1.0,0.0,120.86
50008,5.0,Phone,1,6.0,CC,Male,3.0,3,Mobile,2,Divorced,3,1,16.0,2.0,2.0,0.0,122.93
50009,13.0,Phone,3,9.0,E wallet,Male,2.0,4,Mobile,3,Divorced,2,1,14.0,0.0,1.0,2.0,126.83
50010,2.0,Phone,1,31.0,Debit Card,Male,2.0,5,Mobile,3,Single,2,0,12.0,1.0,1.0,1.0,122.93
50011,4.0,Mobile Phone,1,18.0,Cash on Delivery,Female,2.0,3,Others,3,Divorced,2,0,15.0,9.0,15.0,8.0,295.45
50012,11.0,Mobile Phone,1,6.0,Debit Card,Male,3.0,4,Fashion,3,Single,10,1,13.0,0.0,1.0,0.0,153.81
50013,0.0,Phone,1,11.0,COD,Male,2.0,3,Mobile,3,Single,2,1,13.0,2.0,2.0,2.0,134.41
50014,0.0,Phone,1,15.0,CC,Male,3.0,4,Mobile,3,Divorced,1,1,17.0,0.0,1.0,0.0,133.88
50015,9.0,Mobile Phone,3,15.0,Credit Card,Male,3.0,4,Fashion,2,Single,2,0,16.0,0.0,4.0,7.0,196.19
50016,1.0,Phone,2,12.0,UPI,Male,3.0,3,Mobile,5,Married,5,1,22.0,1.0,1.0,2.0,120.73
50017,0.0,Computer,1,12.0,Debit Card,Female,2.0,4,Mobile,2,Single,2,1,18.0,1.0,1.0,0.0,129.26
50018,0.0,Mobile Phone,3,11.0,E wallet,Male,2.0,4,Laptop & Accessory,3,Single,2,1,11.0,1.0,1.0,3.0,157.44
50019,0.0,Computer,1,13.0,Debit Card,Male,3.0,5,Laptop & Accessory,3,Single,2,1,24.0,1.0,1.0,6.0,160.74
50020,19.0,Mobile Phone,1,20.0,Debit Card,Female,3.0,3,Mobile Phone,4,Divorced,10,1,18.0,1.0,4.0,3.0,149.63
50021,28.0,Mobile Phone,1,7.0,Credit Card,Female,4.0,3,Fashion,5,Married,2,0,19.0,5.0,8.0,10.0,230.50
50022,22.0,Computer,1,9.0,Credit Card,Male,3.0,3,Grocery,4,Married,3,0,14.0,4.0,6.0,5.0,190.20
50023,31.0,Computer,2,8.0,Debit Card,Female,4.0,2,Laptop & Accessory,5,Married,2,0,16.0,6.0,12.0,14.0,265.80
50024,18.0,Mobile Phone,1,10.0,Credit Card,Male,3.0,4,Fashion,4,Married,3,0,13.0,3.0,5.0,7.0,185.00
50025,25.0,Mobile Phone,1,6.0,UPI,Female,3.0,3,Grocery,5,Married,4,0,15.0,4.0,7.0,6.0,210.15`;

/**
 * Generate sample CSV string
 */
export function generateSampleCSV(): string {
  return SAMPLE_CSV_CONTENT;
}

/**
 * Downloads the sample data.csv directly in browser via client-side Blob
 * Zero network dependencies - 100% reliable on any host (Vercel, local, offline)
 */
export function downloadSampleCSV(fileName = "data.csv") {
  const blob = new Blob([SAMPLE_CSV_CONTENT], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", fileName);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
