"""
Enhanced RAG Engine V2 with:
- Hybrid retrieval (BM25 + Vector)
- Conversation memory
- Response caching
- Better error handling
- Performance monitoring
"""
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from dotenv import load_dotenv

# Import our new modules
try:
    from hybrid_retriever import HybridRetriever
    from conversation_memory import ConversationMemory
    from cache_manager import CacheManager
    from error_handler import ErrorHandler, RAGError, APIKeyError, RateLimitError
    from monitoring import PerformanceMonitor
    from performance_optimizer import QueryOptimizer, PerformanceTracker, ConnectionPool
except ImportError:
    from src.hybrid_retriever import HybridRetriever
    from src.conversation_memory import ConversationMemory
    from src.cache_manager import CacheManager
    from src.error_handler import ErrorHandler, RAGError, APIKeyError, RateLimitError
    from src.monitoring import PerformanceMonitor
    from src.performance_optimizer import QueryOptimizer, PerformanceTracker, ConnectionPool

load_dotenv(override=True)


class RAGEngine:
    """
    Enhanced RAG Engine with production-ready features
    """
    
    def __init__(self, data_dir: str = None, api_key: str = None, 
                 enable_cache: bool = True, enable_memory: bool = True):
        """Initialize enhanced RAG engine"""
        # Setup paths
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        self.index_file = data_dir / "index.faiss"
        self.meta_file = data_dir / "meta.json"
        self.emb_file = data_dir / "embeddings.npy"
        
        # API configuration
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Model configuration
        self.model_name = "all-mpnet-base-v2"
        self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        # Use reliable free model as default
        self.default_llm = "meta-llama/llama-3.2-3b-instruct:free"  # Free, fast, reliable
        
        # Confidence thresholds
        self.CONFIDENCE_THRESHOLD = 0.25  # Below this, ask clarifying question
        self.MIN_SIMILARITY = 0.20  # Below this, return "I don't know"
        
        # Load resources
        self._load_resources()
        
        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            embedding_model=self.embedding_model,
            faiss_index=self.index,
            metadata=self.metadata,
            embeddings=self.embeddings
        )
        
        # Initialize conversation memory
        self.enable_memory = enable_memory
        if enable_memory:
            self.memory = ConversationMemory(max_history=10)
        
        # Initialize cache
        self.enable_cache = enable_cache
        if enable_cache:
            self.cache = CacheManager(cache_dir=str(data_dir / "cache"), ttl_hours=24)
        
        # Initialize monitoring
        self.monitor = PerformanceMonitor(db_path=str(data_dir / "monitoring.db"))
        
        # Error handler
        self.error_handler = ErrorHandler()
        
        # Performance optimization
        self.query_optimizer = QueryOptimizer()
        self.perf_tracker = PerformanceTracker()
        self.connection_pool = ConnectionPool(max_connections=100)
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_response_time': 0,
            'total_tokens_used': 0
        }
        
        logger.info("✅ Enhanced RAG Engine initialized successfully with all optimizations")
    
    def _load_resources(self):
        """Load embedding model, FAISS index, and metadata"""
        try:
            # Validate files exist
            if not self.index_file.exists():
                raise FileNotFoundError(f"Index file not found: {self.index_file}")
            if not self.meta_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {self.meta_file}")
            if not self.emb_file.exists():
                raise FileNotFoundError(f"Embeddings file not found: {self.emb_file}")
            
            # Load embedding model
            logger.info(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            
            # Load re-ranker model
            logger.info(f"Loading re-ranker model: {self.reranker_model_name}")
            self.reranker = CrossEncoder(self.reranker_model_name)
            
            # Load FAISS index
            logger.info("Loading FAISS index...")
            self.index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            logger.info("Loading metadata...")
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Load embeddings
            self.embeddings = np.load(str(self.emb_file))
            
            logger.info(f"Loaded {len(self.metadata)} document chunks")
            
        except Exception as e:
            logger.error(f"Error loading resources: {e}")
            raise
    
    def _check_retrieval_confidence(self, chunks: List[Dict], threshold: float = 0.22) -> Tuple[bool, float]:
        """Check if retrieval confidence is above threshold"""
        if not chunks:
            return False, 0.0
        
        # Calculate average similarity score
        scores = [chunk.get('score', 0.0) for chunk in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return avg_score >= threshold, avg_score
    
    def _parse_json_response(self, raw_answer: str) -> Dict[str, Any]:
        """
        Parse JSON block from model response with fallback
        Expects format: answer text + JSON block
        """
        import json
        import re
        
        # Try to find JSON block in response
        json_pattern = r'\{[^{}]*"answer_text"[^{}]*\}'
        matches = re.findall(json_pattern, raw_answer, re.DOTALL)
        
        if matches:
            try:
                # Try to parse the last JSON block found
                parsed = json.loads(matches[-1])
                logger.info("Successfully parsed JSON from response")
                return parsed
            except json.JSONDecodeError:
                logger.warning("Found JSON-like block but failed to parse")
        
        # Fallback: extract text before any JSON attempt
        # Split on common JSON markers
        text_parts = re.split(r'\{[\s\n]*"answer_text"', raw_answer)
        answer_text = text_parts[0].strip() if text_parts else raw_answer
        
        # Return structured fallback
        return {
            "answer_text": answer_text,
            "citations": [],
            "follow_up": None,
            "confidence_estimate": 0.75
        }
    
    def build_prompt(self, question: str, chunks: List[Dict], conversation_context: str = "", 
                    is_low_confidence: bool = False) -> Tuple[str, str]:
        """Build enhanced prompt with enterprise-grade instructions"""
        
        # Enterprise-grade system message
        system_message = """You are an expert enterprise assistant specialized in answering questions from a company's internal documents.

CRITICAL RULES:
1. Use ONLY the provided source passages to answer. If needed information is not present, reply: "I don't know based on provided documents." Do NOT hallucinate.

2. Always cite sources inline for any factual claim using this format: [DOC_NAME, chunk_id] (e.g., [HR_Policy.pdf, chunk_12]). Place citations directly after the sentence that depends on the source.

3. Be concise and clear:
   - For short questions: answer in 1-3 sentences
   - For policy/procedural questions: give a short summary (2-4 sentences) then a bullet list of key steps if the user asked how-to

4. Use a calm professional tone. If user tone is angry, acknowledge emotion briefly ("I understand this is frustrating") then continue professionally.

5. When asked for multi-step instructions, number the steps. When asked for examples, show 1-2 short examples.

6. Limit creative language. Prefer precise, directive language for policy or compliance answers.

7. If asked to provide code, include only working code snippets; keep them minimal and explain assumptions.

8. Return a structured JSON block at the end of your response with fields:
   {
     "answer_text": "main answer here",
     "citations": ["DOC_NAME, chunk_id", ...],
     "follow_up": "optional suggestion",
     "confidence_estimate": 0.95
   }

9. If the question references earlier turn(s), use session memory (if provided) to resolve pronouns; otherwise ask a clarifying question.

10. Respect privacy: do not echo PII from the provided passages unless the user is an authenticated admin.

🚫 ABSOLUTE PROHIBITIONS - NEVER DO THESE:
1. DO NOT mention: "From", "According to", "Source", "Document", "PDF", "Guide", "Section", file names
2. DO NOT copy text directly with phrases like "From Web Development Service Documentation"
3. DO NOT use parenthetical citations like "(Source: filename.pdf, Section X)"
4. DO NOT reference where information came from

✅ WHAT TO DO INSTEAD:
1. READ the information carefully
2. UNDERSTAND what it means
3. WRITE your own clear, natural explanation
4. SPEAK like a knowledgeable colleague, not a document
5. BE conversational and helpful

📝 PERFECT ANSWER EXAMPLE:
"We offer comprehensive software development services! Our team builds custom websites, web applications, and e-commerce platforms. We handle everything from design to deployment, whether you need a simple site or a complex platform."

❌ TERRIBLE ANSWER EXAMPLE:
"From Web Development Service Documentation: 8.1 Web Development Services Overview Unisoftwares builds custom websites... (Source: Company_Overview_Detailed.pdf, Section 2.3)"

REMEMBER: You are NOT a document reader. You are a smart assistant who understands information and explains it naturally!
"""
        
        # Handle low confidence - ask clarifying question
        if is_low_confidence:
            user_message = f"""The retrieved passages appear not to answer the question directly.

**User Question:** {question}

**INSTRUCTION:** BEFORE answering, ask a clarifying question to narrow scope or request permission to escalate. 

Suggested phrasing: "Can you confirm whether you mean X or Y?" or "Would you like me to escalate this to [relevant department]?"

Keep it one short question. Do NOT attempt to answer if you're uncertain."""
            return system_message, user_message
        
        # Build context from chunks with proper citations (truncate if needed)
        context_parts = []
        max_tokens_per_chunk = 300  # Limit chunk size
        
        for i, chunk in enumerate(chunks, 1):
            doc_name = chunk['doc'].split('/')[-1]  # Get filename only
            chunk_id = chunk.get('chunk_id', i)
            
            # Truncate long chunks
            text = chunk['text']
            words = text.split()
            if len(words) > max_tokens_per_chunk:
                text = ' '.join(words[:max_tokens_per_chunk]) + "..."
            
            context_parts.append(f"[{i}] Source: {doc_name} (chunk_{chunk_id})\n{text}")
        
        passages = "\n\n".join(context_parts)
        
        # User message with conversation context
        user_message_parts = []
        
        user_message_parts.append("You are given the following passages (each is labelled). Use ONLY these passages to answer. If no direct answer exists, say \"I don't know based on provided documents.\"\n")
        
        user_message_parts.append(f"**PASSAGES:**\n{passages}\n")
        
        if conversation_context:
            user_message_parts.append(f"\n**Conversation Summary (do not repeat full texts):**\n{conversation_context}\n")
        
        user_message_parts.append(f"\n**QUESTION:** {question}\n")
        
        user_message_parts.append("""
**INSTRUCTIONS:**
- Answer succinctly
- After each factual sentence, add the citation in square brackets [DOC_NAME, chunk_id]
- At the end, provide a JSON block:
{
  "answer_text": "your answer here",
  "citations": [{"doc":"filename.pdf","chunk":12}, ...],
  "follow_up": "optional follow-up suggestion",
  "confidence_estimate": 0.85
}

**Your Response:**""")
        
        user_message = "\n".join(user_message_parts)
        
        return system_message, user_message
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    async def generate_answer(self, system_msg: str, user_msg: str, 
                            model: str = None, temperature: float = 0.0) -> Dict[str, Any]:
        """Generate answer using OpenRouter API with safety controls"""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Please add your API key to .env file")
        
        model = model or self.default_llm
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "RAG Company Chatbot V2"
        }
        
        # Safety controls: temperature=0.0 for policy answers, max_tokens=512 for safety
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": temperature,  # 0.0 for deterministic policy answers
            "max_tokens": 512  # Limit generation for safety
        }
        
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                logger.info(f"Calling OpenRouter API with model: {model}")
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 401:
                    error_msg = "Invalid API key. Please check your OPENROUTER_API_KEY in .env file"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                elif response.status_code == 429:
                    error_msg = "Rate limit exceeded. Please wait a moment and try again"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                elif response.status_code != 200:
                    error_detail = f"OpenRouter API error: {response.status_code} - {response.text}"
                    logger.error(error_detail)
                    raise Exception(error_detail)
                
                data = response.json()
                logger.info("Successfully received response from OpenRouter")
                
                raw_answer = data["choices"][0]["message"]["content"]
                
                # Parse JSON block from answer (if present)
                parsed_data = self._parse_json_response(raw_answer)
                
                return {
                    "answer": parsed_data.get("answer_text", raw_answer),
                    "model_used": model,
                    "tokens_used": data.get("usage", {}),
                    "parsed_json": parsed_data
                }
                
        except httpx.TimeoutException:
            error_msg = "Request timeout - OpenRouter API is taking too long. Please try again"
            logger.error(error_msg)
            raise Exception(error_msg)
        except httpx.ConnectError as e:
            error_msg = "Cannot connect to OpenRouter API. Please check your internet connection"
            logger.error(f"{error_msg}: {e}")
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Error calling OpenRouter: {e}")
            raise
    
    async def query(self, question: str, top_k: int = 5, 
                   model: str = None, temperature: float = 0.7,
                   session_id: str = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Enhanced query with caching, conversation memory, and monitoring
        """
        start_time = time.time()
        self.metrics['total_queries'] += 1
        error_type = None
        success = False
        tokens_used = 0
        cache_hit = False
        
        try:
            # Check cache first
            if use_cache and self.enable_cache:
                cached = self.cache.get(question, model)
                if cached:
                    self.metrics['cache_hits'] += 1
                    logger.info(f"✅ Cache hit for query: {question[:50]}...")
                    return cached
            
            # Get conversation context if memory enabled
            conversation_context = ""
            if self.enable_memory and session_id:
                conversation_context = self.memory.get_context(session_id, include_last_n=2)
                # Enhance query with context
                enhanced_question = self.memory.get_contextual_query(session_id, question)
            else:
                enhanced_question = question
            
            # Step 1: Hybrid retrieval with performance tracking
            logger.info(f"Retrieving top {top_k} chunks for: {enhanced_question[:100]}...")
            retrieval_start = time.time()
            chunks = self.hybrid_retriever.retrieve(enhanced_question, top_k=top_k, alpha=0.6)
            retrieval_time = time.time() - retrieval_start
            self.perf_tracker.record_timing('retrieval', retrieval_time)
            
            if not chunks:
                return {
                    "success": False,
                    "answer": "No relevant information found in the documents.",
                    "sources": [],
                    "retrieved_chunks": [],
                    "execution_time": time.time() - start_time,
                    "error": "No relevant chunks found"
                }
            
            # Step 1.5: Check retrieval confidence
            is_confident, avg_score = self._check_retrieval_confidence(chunks, threshold=0.22)
            logger.info(f"Retrieval confidence: {avg_score:.3f} (threshold: 0.22)")
            
            if not is_confident:
                logger.warning(f"Low retrieval confidence ({avg_score:.3f}). Returning fallback.")
                return {
                    "success": False,
                    "answer": "I don't know based on provided documents. The retrieved information doesn't seem directly relevant to your question. Would you like me to search more broadly or escalate this query?",
                    "sources": [],
                    "retrieved_chunks": chunks,
                    "execution_time": time.time() - start_time,
                    "error": "Low retrieval confidence",
                    "confidence_score": avg_score
                }
            
            # Step 2: Build prompt with conversation context
            system_msg, user_msg = self.build_prompt(question, chunks, conversation_context, is_low_confidence=False)
            
            # Step 3: Generate answer with performance tracking
            logger.info(f"Generating answer using {model or self.default_llm}...")
            generation_start = time.time()
            generation_result = await self.generate_answer(
                system_msg, user_msg, model, temperature
            )
            generation_time = time.time() - generation_start
            self.perf_tracker.record_timing('generation', generation_time)
            
            # Step 4: Prepare response
            sources = []
            for i, chunk in enumerate(chunks, 1):
                sources.append({
                    "source_number": i,
                    "doc": chunk["doc"],
                    "chunk_id": chunk["chunk_id"],
                    "relevance": chunk["relevance"],
                    "score": chunk["score"],
                    "text_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                })
            
            execution_time = time.time() - start_time
            self.perf_tracker.record_timing('total', execution_time)
            
            # Extract confidence from parsed JSON or use retrieval score
            parsed_json = generation_result.get("parsed_json", {})
            confidence_score = parsed_json.get("confidence_estimate", avg_score)
            
            result = {
                "success": True,
                "answer": generation_result["answer"],
                "sources": sources,
                "retrieved_chunks": chunks,
                "model_used": generation_result["model_used"],
                "tokens_used": generation_result.get("tokens_used", {}),
                "execution_time": execution_time,
                "top_k": top_k,
                "temperature": temperature,
                "from_cache": False,
                "confidence_score": confidence_score,
                "performance": {
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time
                }
            }
            
            # Cache the result
            if use_cache and self.enable_cache:
                self.cache.set(question, result, model)
            
            # Add to conversation memory
            if self.enable_memory and session_id:
                self.memory.add_turn(session_id, question, result["answer"], sources)
            
            # Update metrics
            self.metrics['avg_response_time'] = (
                (self.metrics['avg_response_time'] * (self.metrics['total_queries'] - 1) + execution_time) 
                / self.metrics['total_queries']
            )
            if 'total_tokens' in generation_result.get("tokens_used", {}):
                tokens_used = generation_result["tokens_used"]["total_tokens"]
                self.metrics['total_tokens_used'] += tokens_used
            
            # Log to monitoring
            success = True
            self.monitor.log_request(
                question=question,
                response_time=execution_time,
                success=True,
                cache_hit=cache_hit,
                tokens_used=tokens_used
            )
            
            return result
            
        except Exception as e:
            # Handle error with error handler
            error_response = self.error_handler.handle_error(e, "query_pipeline")
            error_type = error_response.get('error_code', 'UNKNOWN_ERROR')
            
            # Log error to monitoring
            self.monitor.log_request(
                question=question,
                response_time=time.time() - start_time,
                success=False,
                error_type=error_type,
                cache_hit=cache_hit,
                tokens_used=0
            )
            
            logger.error(f"Error in query pipeline: {e}")
            return {
                "success": False,
                "answer": error_response.get('user_message', f"An error occurred: {str(e)}"),
                "sources": [],
                "retrieved_chunks": [],
                "execution_time": time.time() - start_time,
                "error": str(e),
                "error_code": error_type
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics with detailed timing stats"""
        cache_stats = self.cache.get_stats() if self.enable_cache else {}
        perf_stats = self.perf_tracker.get_stats()
        bottlenecks = self.perf_tracker.identify_bottlenecks()
        
        return {
            **self.metrics,
            'cache_hit_rate': f"{(self.metrics['cache_hits'] / max(self.metrics['total_queries'], 1) * 100):.1f}%",
            'cache_stats': cache_stats,
            'performance_stats': perf_stats,
            'bottlenecks': bottlenecks
        }
    
    def export_conversation(self, session_id: str, format: str = 'markdown') -> str:
        """Export conversation history"""
        if not self.enable_memory:
            return "Conversation memory is disabled"
        
        return self.memory.export_conversation(session_id, format)
    
    def clear_cache(self):
        """Clear response cache"""
        if self.enable_cache:
            self.cache.clear_all()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "embedding_model": self.model_name,
            "total_chunks": len(self.metadata),
            "documents": list(set([chunk['doc'] for chunk in self.metadata])),
            "index_size": self.index.ntotal if hasattr(self.index, 'ntotal') else 0
        }
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available LLM models"""
        return [
            {
                "id": "meta-llama/llama-3.2-3b-instruct:free",
                "name": "Llama 3.2 3B (Recommended - Free & Fast)",
                "provider": "Meta"
            },
            {
                "id": "openai/gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo (Balanced - Paid)",
                "provider": "OpenAI"
            },
            {
                "id": "google/gemini-pro",
                "name": "Gemini Pro (High Quality - Paid)",
                "provider": "Google"
            },
            {
                "id": "anthropic/claude-3-haiku",
                "name": "Claude 3 Haiku (Best Quality - Paid)",
                "provider": "Anthropic"
            },
            {
                "id": "mistralai/mistral-7b-instruct:free",
                "name": "Mistral 7B (Free - Good Quality)",
                "provider": "Mistral"
            },
            {
                "id": "meta-llama/llama-3.1-8b-instruct:free",
                "name": "Llama 3.1 8B (Free - Better Quality)",
                "provider": "Meta"
            }
        ]
