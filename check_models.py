import boto3

client = boto3.client("bedrock", region_name="us-east-1")
resp = client.list_foundation_models()
models = resp["modelSummaries"]
print(f"Total models returned: {len(models)}\n")
for m in models[:40]:
    print(m["modelId"])