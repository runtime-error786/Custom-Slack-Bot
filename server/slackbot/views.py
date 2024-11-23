from rest_framework.decorators import api_view
from rest_framework.response import Response
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import requests
import json

# Slack bot token
slack_token = ""
client = WebClient(token=slack_token)

@api_view(['POST'])
def slack_event(request):
    try:
        slack_event = json.loads(request.body)

        # Check if the request contains a challenge (for URL verification)
        if "challenge" in slack_event:
            return Response({"challenge": slack_event["challenge"]})

        # Send a quick acknowledgment to Slack
        if "event" in slack_event:
            event = slack_event["event"]
            # Process the event asynchronously
            import threading

            def process_event(event_data):
                try:
                    if "text" in event_data:
                        question = event_data["text"]
                        channel_id = event_data["channel"]

                        # Send the question to your Django API
                        response = requests.post(
    "http://127.0.0.1:3001/chat/",
    json={"question": question},
    headers={"Content-Type": "application/json"},
    verify=False  # Disable SSL verification
)


                        if response.status_code == 200:
                            data = response.json()
                            answer = data.get("response", "Sorry, I couldn't process the request.")
                        else:
                            answer = "There was an error processing your question."

                        # Send the response back to Slack
                        client.chat_postMessage(channel=channel_id, text=f"Answer: {answer}")
                except Exception as e:
                    print(f"Error processing event: {e}")

            # Run the event processing in a new thread
            threading.Thread(target=process_event, args=(event,)).start()

        return Response({"status": "ok"})

    except json.JSONDecodeError:
        return Response({"error": "Invalid JSON format"}, status=400)

    except requests.exceptions.RequestException as e:
        return Response({"error": f"Error communicating with API: {str(e)}"}, status=500)

    except Exception as e:
        return Response({"error": f"Unexpected error: {str(e)}"}, status=500)
