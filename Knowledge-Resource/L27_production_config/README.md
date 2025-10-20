# Lesson 27: Production config & K8s manifests (vLLM, TRT-LLM, gateway, RAG, tools)


1) app configs (staging/prod)
```
/app/configs/staging.yaml

env: "staging"
models:
  primary: { name: "my-minilm", engine: "vllm", base_url: "http://vllm:8001", max_ctx: 32768 }
  lowlat:  { name: "my-minilm", engine: "trt",  base_url: "http://trt:8002",  max_out: 256 }
  small:   { name: "my-small",  engine: "vllm", base_url: "http://small:8001" }
decoding: { temperature: 0.8, top_p: 0.9, rep_penalty: 1.1, stop: [] }
routing:
  rules:
    - match: { domain: "rag_long" }     -> primary
    - match: { domain: "chat_short" }   -> lowlat
    - match: { uncertainty_gt: 0.5 }    -> primary
budgets:  { max_prompt: 6000, max_new: 512 }
rates:    { per_ip_min: 60, per_key_min: 120 }
safety:   { pii_redact: true, citation_required_rag: true }
rag:
  base_url: "http://rag:8080"
  alpha: 0.6
  top_k: 20
  top_n: 6
observability:
  prometheus_path: "/metrics"
  otlp_endpoint: "http://otel-collector:4318"
secrets_refs:
  redis_url: "redis://redis:6379/0"
  s3_bucket: "YOUR_BUCKET"
  s3_region: "YOUR_REGION"
  s3_prefix: "corpora/v1"


/app/configs/prod.yaml (same keys; tweak limits, add stricter budgets & rates).
```
2) Kubernetes: namespaces, secrets, config
2.1 namespace & RBAC
```
apiVersion: v1
kind: Namespace
metadata: { name: sunforge-llm }

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata: { name: read-config, namespace: sunforge-llm }
rules:
- apiGroups: [""]
  resources: ["configmaps","secrets"]
  verbs: ["get","list"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata: { name: read-config-binding, namespace: sunforge-llm }
subjects:
- kind: ServiceAccount
  name: default
  namespace: sunforge-llm
roleRef:
  kind: Role
  name: read-config
  apiGroup: rbac.authorization.k8s.io
```
2.2 secrets (env-only; no raw keys in images)
```
apiVersion: v1
kind: Secret
metadata: { name: app-secrets, namespace: sunforge-llm }
type: Opaque
stringData:
  OPENAI_API_KEY: "YOUR_OPTIONAL_KEY"
  S3_ACCESS_KEY: "YOUR_KEY"
  S3_SECRET_KEY: "YOUR_SECRET"
  REDIS_URL: "redis://redis:6379/0"
```
2.3 configmap (switch staging/prod by APP_ENV)
```
apiVersion: v1
kind: ConfigMap
metadata: { name: app-config, namespace: sunforge-llm }
data:
  APP_ENV: "staging"
```
3) vLLM deployment (single-GPU pod, paged KV, streaming)
```
apiVersion: apps/v1
kind: Deployment
metadata: { name: vllm, namespace: sunforge-llm, labels: { app: vllm } }
spec:
  replicas: 1
  selector: { matchLabels: { app: vllm } }
  template:
    metadata:
      labels: { app: vllm }
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8001"
        prometheus.io/path: "/metrics"
    spec:
      nodeSelector: { "nvidia.com/gpu.present": "true" }
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - "--model=/models/my-minilm"
          - "--dtype=bfloat16"
          - "--host=0.0.0.0"
          - "--port=8001"
          - "--tensor-parallel-size=1"
          - "--max-model-len=32768"
          - "--gpu-memory-utilization=0.90"
        ports: [{ containerPort: 8001 }]
        resources:
          limits: { "nvidia.com/gpu": 1, cpu: "2", memory: "16Gi" }
          requests: { "nvidia.com/gpu": 1, cpu: "1", memory: "12Gi" }
        volumeMounts:
          - { name: model-vol, mountPath: /models, readOnly: true }
      volumes:
        - name: model-vol
          persistentVolumeClaim: { claimName: hf-models-pvc }
---
apiVersion: v1
kind: Service
metadata: { name: vllm, namespace: sunforge-llm }
spec:
  selector: { app: vllm }
  ports: [{ name: http, port: 8001, targetPort: 8001 }]


Put your HuggingFace-style folder into a ReadOnlyMany PVC hf-models-pvc or bake a separate model image.
```
4) TensorRT-LLM (low-latency engine)
```
apiVersion: apps/v1
kind: Deployment
metadata: { name: trt, namespace: sunforge-llm, labels: { app: trt } }
spec:
  replicas: 1
  selector: { matchLabels: { app: trt } }
  template:
    metadata:
      labels: { app: trt }
    spec:
      nodeSelector: { "nvidia.com/gpu.present": "true" }
      containers:
      - name: trt
        image: nvcr.io/nvidia/tritonserver:23.10-py3   # or TRT-LLM server image you built
        command: ["bash","-lc"]
        args:
          - >
            tensorrt_llm_server
            --engine_dir /engines/my-minilm
            --port 8002
            --tp_size 1
            --paged_kv_cache
            --enable_chunked_prefill
        ports: [{ containerPort: 8002 }]
        resources:
          limits: { "nvidia.com/gpu": 1, cpu: "2", memory: "14Gi" }
          requests:{ "nvidia.com/gpu": 1, cpu: "1", memory: "10Gi" }
        volumeMounts:
          - { name: trt-engine, mountPath: /engines, readOnly: true }
      volumes:
        - name: trt-engine
          persistentVolumeClaim: { claimName: trt-engine-pvc }
---
apiVersion: v1
kind: Service
metadata: { name: trt, namespace: sunforge-llm }
spec:
  selector: { app: trt }
  ports: [{ name: http, port: 8002, targetPort: 8002 }]
```
5) “small” vLLM (draft/fallback)

Duplicate the vLLM manifest as small with a smaller model path and lower memory requests.

6) RAG service (CPU or 1 tiny GPU)
```
apiVersion: apps/v1
kind: Deployment
metadata: { name: rag, namespace: sunforge-llm, labels: { app: rag } }
spec:
  replicas: 1
  selector: { matchLabels: { app: rag } }
  template:
    metadata: { labels: { app: rag } }
    spec:
      containers:
      - name: rag
        image: YOUR_REGISTRY/rag-service:latest
        env:
          - { name: S3_BUCKET, valueFrom: { secretKeyRef: { name: app-secrets, key: S3_BUCKET }}}
          - { name: S3_REGION, value: "YOUR_REGION" }
        ports: [{ containerPort: 8080 }]
        readinessProbe: { httpGet: { path: "/health", port: 8080 }, initialDelaySeconds: 5, periodSeconds: 10 }
        resources:
          requests: { cpu: "500m", memory: "1Gi" }
          limits:   { cpu: "1",    memory: "2Gi" }
        volumeMounts:
          - { name: rag-index, mountPath: /indexes, readOnly: true }
      volumes:
        - name: rag-index
          persistentVolumeClaim: { claimName: rag-index-pvc }
---
apiVersion: v1
kind: Service
metadata: { name: rag, namespace: sunforge-llm }
spec:
  selector: { app: rag }
  ports: [{ name: http, port: 8080, targetPort: 8080 }]
```
7) Tools runner (sandboxed; no egress)
```
apiVersion: apps/v1
kind: Deployment
metadata: { name: tools, namespace: sunforge-llm, labels: { app: tools } }
spec:
  replicas: 1
  selector: { matchLabels: { app: tools } }
  template:
    metadata: { labels: { app: tools } }
    spec:
      containers:
      - name: tools
        image: YOUR_REGISTRY/tools-runner:latest
        securityContext:
          runAsNonRoot: true
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
        ports: [{ containerPort: 7070 }]
        env:
          - { name: NO_NET, value: "1" }
        resources:
          requests: { cpu: "500m", memory: "1Gi" }
          limits:   { cpu: "1",    memory: "2Gi" }
      # Block egress
      dnsPolicy: ClusterFirst
      hostNetwork: false
---
apiVersion: v1
kind: Service
metadata: { name: tools, namespace: sunforge-llm }
spec:
  selector: { app: tools }
  ports: [{ name: http, port: 7070, targetPort: 7070 }]
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: tools-deny-egress, namespace: sunforge-llm }
spec:
  podSelector: { matchLabels: { app: tools } }
  policyTypes: ["Egress","Ingress"]
  egress: []   # deny all
  ingress:
  - from:
    - podSelector: { matchLabels: { app: gateway } }
```
8) Gateway (FastAPI) with canary, budgets, redaction, metrics
```
apiVersion: apps/v1
kind: Deployment
metadata: { name: gateway, namespace: sunforge-llm, labels: { app: gateway } }
spec:
  replicas: 2
  selector: { matchLabels: { app: gateway } }
  template:
    metadata:
      labels: { app: gateway }
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8081"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: gateway
        image: YOUR_REGISTRY/gateway:latest
        envFrom:
          - secretRef: { name: app-secrets }
          - configMapRef: { name: app-config }
        env:
          - { name: VLLM_URL, value: "http://vllm:8001" }
          - { name: TRT_URL,  value: "http://trt:8002" }
          - { name: SMALL_URL,value: "http://small:8001" }
          - { name: RAG_URL,  value: "http://rag:8080" }
        ports: [{ containerPort: 8081 }]
        readinessProbe: { httpGet: { path: "/health", port: 8081 }, initialDelaySeconds: 5, periodSeconds: 10 }
        resources:
          requests: { cpu: "500m", memory: "1Gi" }
          limits:   { cpu: "1",    memory: "2Gi" }
        # HPA-friendly
        lifecycle:
          preStop: { exec: { command: ["sh","-c","sleep 5"] } }
---
apiVersion: v1
kind: Service
metadata: { name: gateway, namespace: sunforge-llm }
spec:
  selector: { app: gateway }
  ports: [{ name: http, port: 80, targetPort: 8081 }]
```
8.1 Ingress (TLS, sticky cookies for A/B)
```
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api
  namespace: sunforge-llm
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/affinity: "cookie"
    nginx.ingress.kubernetes.io/session-cookie-name: "exp_bucket"
spec:
  tls:
  - hosts: ["api.YOUR_DOMAIN"]
    secretName: tls-cert
  rules:
  - host: api.YOUR_DOMAIN
    http:
      paths:
      - path: /
        pathType: Prefix
        backend: { service: { name: gateway, port: { number: 80 } } }
```
9) autoscaling & quotas
9.1 HPA for gateway
```
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: gateway-hpa, namespace: sunforge-llm }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: gateway }
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } }


For vLLM/TRT pods, prefer vertical tuning and engine-level continuous batching; scale replica count if you have many GPUs.
```
9.2 ResourceQuota (keep costs sane in staging)
```
apiVersion: v1
kind: ResourceQuota
metadata: { name: rq, namespace: sunforge-llm }
spec:
  hard:
    requests.cpu: "8"
    requests.memory: "32Gi"
    limits.nvidia.com/gpu: "2"
```
10) observability plumbing

- Prometheus: scrape gateway, vllm (if exposed), plus node exporters; add alerts:
- p95 TTFT > SLO for 5 min
- 5xx > 1% for 2 min
- GPU OOM logs detected

OpenTelemetry: set OTEL_EXPORTER_OTLP_ENDPOINT in gateway; trace generate, rag_answer, engine calls.

11) security: network policies & egress control
```
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: default-deny, namespace: sunforge-llm }
spec:
  podSelector: {}
  policyTypes: ["Ingress","Egress"]
  ingress: []   # deny by default
  egress: []    # deny by default
---
# allow ingress to gateway from ingress-controller
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: allow-ingress-to-gateway, namespace: sunforge-llm }
spec:
  podSelector: { matchLabels: { app: gateway } }
  ingress:
  - from:
    - namespaceSelector: { matchLabels: { name: ingress-nginx } }


Add specific egress allows for gateway → vllm/trt/rag/tools and rag → S3 only.
```
12) storage PVC examples
```
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: hf-models-pvc, namespace: sunforge-llm }
spec:
  accessModes: ["ReadOnlyMany"]
  storageClassName: YOUR_RO_STORAGECLASS
  resources: { requests: { storage: 50Gi } }

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: trt-engine-pvc, namespace: sunforge-llm }
spec:
  accessModes: ["ReadOnlyMany"]
  storageClassName: YOUR_RO_STORAGECLASS
  resources: { requests: { storage: 30Gi } }

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: rag-index-pvc, namespace: sunforge-llm }
spec:
  accessModes: ["ReadOnlyMany"]
  storageClassName: YOUR_RO_STORAGECLASS
  resources: { requests: { storage: 20Gi } }
```
13) smoke tests you should run immediately
```
kubectl -n sunforge-llm port-forward svc/gateway 8081:80 then:

GET /health → 200

GET /metrics → Prom metrics visible

/v1/chat/completions with temperature=0 → deterministic hash

/rag_answer with a seeded doc → requires [1] citation

Check GPU visibility in vLLM/TRT logs (nvidia-smi inside pod if needed).
```
14) rollout steps (staging → canary)
```
kubectl apply -n sunforge-llm -f manifests/ (namespace→secrets→pvc→services→deploys→ingress).

Wait for READY pods; tail logs for vLLM/TRT warmup.

Run smoke tests; warm caches with 100 synthetic requests.

Switch APP_ENV=prod in app-config (or deploy prod.yaml baked into gateway env).

Enable canary at the gateway router (10% traffic to primary if changing models).

Watch dashboards/alerts; promote if green.
```
15) quick knobs to tune
```
vLLM:

--max-model-len (must match your RoPE scaling)

--gpu-memory-utilization (0.85–0.92 sweet spot)

--max-num-seqs / --max-num-batched-tokens for bursty loads

TRT-LLM:

--paged_kv_cache, --enable_chunked_prefill

TP size (2 on dual-GPU nodes), weight/kv quant flags

Gateway:

rate limits, token budgets, decoding defaults (lesson 18)

streaming on by default
```
16) your actions (right now)

 - Replace placeholders (bucket/region/domain) in the YAML above.
 - Apply namespace, secrets, configmaps; create PVCs (or bake models into images).
 - Deploy vLLM, TRT-LLM, RAG, tools, then gateway + ingress.
 - Run the smoke test checklist; post any failing step and logs, and I’ll help debug.
 - Once stable in staging, flip to a prod canary with alerts on.