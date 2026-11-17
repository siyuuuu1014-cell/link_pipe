export interface Detection {
  class_id: number
  class_name: 'crack' | 'water' | 'leak'
  class_name_zh: string
  confidence: number
  bbox_xyxy: number[]
  bbox_normalized: number[]
  area_ratio: number
  location_text: string
}

export interface DetectionResult {
  record_id: number
  status: string
  status_zh: string
  risk_level: 'high' | 'medium' | 'none'
  risk_level_zh: string
  counts: Record<string, number>
  detections: Detection[]
  summary: string
  advice: string[]
  original_image_url: string
  result_image_url: string
  report_url: string
  inference_ms: number
  disclaimer: string
}

export interface HistoryRecord {
  id: number
  created_at: string
  source_type: string
  original_filename: string
  status: string
  risk_level: string
  summary: string
  detections: Detection[]
  advice: string[]
  model_version: string
  inference_ms: number
  original_image_url?: string
  result_image_url?: string
  video_result_url?: string
  report_url: string
}

export interface ModelMetrics {
  precision: number
  recall: number
  map50: number
  map50_95: number
  per_class_map50_95: Record<string, number | null>
  speed_ms: Record<string, number>
  model_size_mb: number
}
