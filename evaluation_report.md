# Operational Analysis

- **Model used**: gemini-3.1-flash-lite  
- **Test set**: 44 claims → 44 successful API calls (1 retry for 503 error, handled).  
- **Total input tokens**: 122,943  
- **Total output tokens**: 13,424  
- **Estimated cost**: ~$0.013 (using $0.000075/1k input, $0.0003/1k output)  
- **Average latency**: ~3.5s per call, total runtime ~15 min (with 10s delay between calls).  
- **Rate limit strategy**: 10s fixed delay + exponential backoff (5,10,20s) to stay under 60 RPM.