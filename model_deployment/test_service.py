import bentoml

SERVICE_URL = "http://localhost:3000"

def main():
    with bentoml.SyncHTTPClient(SERVICE_URL) as client: 
        result = client.predict(7)
        print(result)

if __name__ == "__main__":
    main()

print('hello world')