export interface RestaurantResult {
  business_id: string;
  name: string;
  city: string;
  stars: number;
  review_count: number;
  categories: string;
  address: string;
  similarity_score: number;
  snippet: string;
}

export interface SearchResponse {
  query: string;
  results: RestaurantResult[];
  latency_ms: number;
  total_results: number;
}
