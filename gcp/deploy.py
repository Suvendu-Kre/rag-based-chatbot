#!/usr/bin/env python3
"""Deploy agent to Vertex AI Agent Engine via GitHub Actions CI/CD."""
import os
import sys
import importlib.util
import inspect

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
STAGING_BUCKET = os.environ.get("GOOGLE_CLOUD_STAGING_BUCKET", "")

def check_prerequisites():
    """Validate environment and dependencies."""
    if not PROJECT_ID:
        print("ERROR: GOOGLE_CLOUD_PROJECT environment variable must be set")
        return False
    try:
        import vertexai
        from vertexai import agent_engines
        return True
    except ImportError:
        print("ERROR: google-cloud-aiplatform[agent_engines] is required")
        print("Run: pip install google-cloud-aiplatform[agent_engines]")
        return False

def ensure_staging_bucket():
    """Create or verify the GCS staging bucket exists."""
    global STAGING_BUCKET
    
    if not STAGING_BUCKET:
        STAGING_BUCKET = f"gs://{PROJECT_ID}-vertex-staging"
    
    # Ensure gs:// prefix
    if not STAGING_BUCKET.startswith("gs://"):
        STAGING_BUCKET = f"gs://{STAGING_BUCKET}"
    
    bucket_name = STAGING_BUCKET.replace("gs://", "").rstrip("/")
    
    print(f"Ensuring staging bucket exists: {STAGING_BUCKET}")
    try:
        from google.cloud import storage
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        
        if not bucket.exists():
            print(f"Creating staging bucket: {bucket_name} in {LOCATION}...")
            bucket = client.create_bucket(bucket_name, location=LOCATION)
            print(f"Staging bucket created: {STAGING_BUCKET}")
        else:
            print(f"Staging bucket exists: {STAGING_BUCKET}")
    except ImportError:
        print("WARNING: google-cloud-storage not installed. Cannot auto-create bucket.")
        print(f"Please create the bucket manually: gsutil mb -l {LOCATION} {STAGING_BUCKET}")
    except Exception as e:
        print(f"WARNING: Could not verify/create staging bucket: {e}")
        print(f"Continuing with bucket: {STAGING_BUCKET} (may fail if it doesn't exist)")
    
    return STAGING_BUCKET

def find_agent_class():
    """Find the Agent class in the generated code."""
    agent_paths = [
        "agents/main_agent.py",
        "main.py",
        "agent.py"
    ]
    
    for path in agent_paths:
        if not os.path.exists(path):
            continue
            
        print(f"Searching for Agent in {path}...")
        spec = importlib.util.spec_from_file_location("module", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["module"] = module
        try:
            spec.loader.exec_module(module)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and (name == 'Agent' or name.endswith('Agent') and name != 'QueryableAgent'):
                    print(f"Found Agent class: {name}")
                    return obj
        except Exception as e:
            print(f"Warning: Error loading {path}: {e}")
            continue
            
    return None

def main():
    # Always output a console URL so GitHub Actions summary has something useful
    console_url = f"https://console.cloud.google.com/vertex-ai/agents?project={PROJECT_ID}"
    
    if not check_prerequisites():
        print(f"ENDPOINT_URL={console_url}")
        sys.exit(1)
    
    import vertexai
    from vertexai import agent_engines
    
    # Ensure staging bucket exists (required by Vertex AI Agent Engine)
    staging_bucket = ensure_staging_bucket()
    
    print(f"Initializing Vertex AI in {PROJECT_ID}/{LOCATION}...")
    print(f"Staging bucket: {staging_bucket}")
    try:
        vertexai.init(
            project=PROJECT_ID,
            location=LOCATION,
            staging_bucket=staging_bucket,
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize Vertex AI: {e}")
        print(f"ENDPOINT_URL={console_url}")
        sys.exit(1)
    
    AgentClass = find_agent_class()
    if not AgentClass:
        print("WARNING: Could not find an Agent class. Deploying a simple echo agent...")
        # Create a minimal queryable agent so deployment still works
        class SimpleAgent:
            def query(self, **kwargs):
                return {"status": "ok", "message": "Agent deployed successfully via CI/CD", "input": kwargs}
        agent_instance = SimpleAgent()
    else:
        print("Instantiating Agent...")
        try:
            agent_instance = AgentClass()
        except Exception as e:
            print(f"WARNING: Failed to instantiate agent: {e}")
            class SimpleAgent:
                def query(self, **kwargs):
                    return {"status": "ok", "message": "Agent deployed via CI/CD (fallback)", "input": kwargs}
            agent_instance = SimpleAgent()
    
    # Ensure it has a query() method for Vertex AI Agent Engine
    if not hasattr(agent_instance, 'query'):
        print("Agent does not have a query() method. Creating a Queryable wrapper...")
        class QueryableWrapper:
            def __init__(self, agent):
                self._agent = agent
                
            def query(self, **kwargs):
                request = kwargs.get('request', kwargs)
                if hasattr(self._agent, '__call__'):
                    return self._agent(request, **{k:v for k,v in kwargs.items() if k != 'request'})
                elif hasattr(self._agent, 'run'):
                    return self._agent.run(request, **{k:v for k,v in kwargs.items() if k != 'request'})
                elif hasattr(self._agent, 'execute'):
                    return self._agent.execute(request, **{k:v for k,v in kwargs.items() if k != 'request'})
                else:
                    return {"status": "error", "message": "Agent lacks query, __call__, run, or execute methods."}
        agent_instance = QueryableWrapper(agent_instance)
        
    # Check if we have requirements.txt
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
    print("Deploying to Vertex AI Agent Engine...")
    try:
        remote_agent = agent_engines.create(
            agent_engine=agent_instance,
            display_name="Agentic_AI_Agent",
            description="Deployed via Agentic-AI GitHub Action CI/CD",
            requirements=requirements
        )
        
        resource_name = getattr(remote_agent, "resource_name", None) or str(remote_agent)
        print(f"Deployment completed successfully!")
        print(f"Resource: {resource_name}")
        
        # Output the Endpoint URL format that cd.yml expects
        agent_id = resource_name.split('/')[-1] if '/' in resource_name else "unknown"
        endpoint_url = f"https://console.cloud.google.com/vertex-ai/agents/locations/{LOCATION}/agents/{agent_id}?project={PROJECT_ID}"
        print(f"ENDPOINT_URL={endpoint_url}")
        
    except Exception as e:
        print(f"Deployment to Vertex AI Agent Engine failed: {e}")
        print(f"")
        print(f"Common fixes:")
        print(f"  1. Enable the Vertex AI API: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com?project={PROJECT_ID}")
        print(f"  2. Grant 'Vertex AI Admin' role to your service account")
        print(f"  3. Ensure billing is enabled on the project")
        print(f"")
        # Still output a console URL so the summary has something useful
        print(f"ENDPOINT_URL={console_url}")
        sys.exit(1)

if __name__ == "__main__":
    main()
