// src/app/api/pipeline.ts

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface DistilBertResult {
  answer: string;
  answerable: boolean;
  start_token: number;
  end_token: number;
  start_logit: number;
  end_logit: number;
  cls_logit: number;
  null_score: number;
  tokens: string[];
  token_type: string[];
  start_logits_top5: [number, number][];
  end_logits_top5: [number, number][];
  inference_time_ms?: number;
}

export interface BiLSTMResult {
  answer: string;
  answerable: boolean;
  start_token: number;
  end_token: number;
  start_logit: number;
  end_logit: number;
  answerable_score: number;
  ctx_tokens: string[];
  q_tokens: string[];
  attention_weights: number[][];  // shape ≤8×≤8
  inference_time_ms?: number;
}

export interface SummarizationResult {
  sentences: string[];
  selected_indices: number[];
  summary: string;
  scores: number[];
  total: number;
  kept: number;
  compression: number;
  tfidf_scores: number[];
  textrank_scores: number[];
  position_scores: number[];
}

export interface RetrievedPassage {
  rank: number;
  passage_id: number;
  passage: string;
  score: number;
  tfidf_score: number;
  bm25_score: number;
  summary: SummarizationResult;
  qa_distilbert?: DistilBertResult;
  qa_bilstm?: BiLSTMResult;
}

export interface PipelineResponse {
  query: string;
  retrieval_time_ms: number;
  summarization_time_ms: number;
  qa_time_ms?: number;
  total_time_ms: number;
  retrieved: RetrievedPassage[];
  best_qa?: DistilBertResult;   // best answer across all retrieved passages
}

export async function runPipeline(query: string, topK = 3): Promise<PipelineResponse> {
  const res = await fetch(`${BASE_URL}/api/pipeline`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) throw new Error(`Pipeline API error: ${res.status}`);
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function runDistilBertQA(
  question: string,
  context: string
): Promise<DistilBertResult & { inference_time_ms: number }> {
  const res = await fetch(`${BASE_URL}/api/qa/distilbert`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, context }),
  });
  if (!res.ok) throw new Error(`DistilBERT API error: ${res.status}`);
  return res.json();
}

export async function runBiLSTMQA(
  question: string,
  context: string
): Promise<BiLSTMResult & { inference_time_ms: number }> {
  const res = await fetch(`${BASE_URL}/api/qa/bilstm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, context }),
  });
  if (!res.ok) throw new Error(`BiLSTM API error: ${res.status}`);
  return res.json();
}
