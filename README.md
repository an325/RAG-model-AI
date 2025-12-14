# RAG-model-AI
ollama rag model AI for sos report read

docker build -t sosreport-ui .
docker images
docker run -it --rm   -p 7860:7860   -e OLLAMA_HOST=172.22.31.128:11434 sosreport-ui
docker exec -it b9636ae8a0b5 bash
----
ollama install on local 
#curl -fsSL https://ollama.com/install.sh | sh
#ollama run llama3 or ollama run mistral
# ollama serve
edit ollama.service and add below line 
#Environment="OLLAMA_HOST=0.0.0.0:11434"
#systemct restart ollama

-----
http://localhost:7860
