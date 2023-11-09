+++
title = "Llm_bot"
date = "2023-11-08T20:36:14-06:00"
author = "Elliott Davis"
cover = "images/robot_finger.gif"
tags = ["llm"]
keywords = ["llm", "slack"]
description = ""
showFullContent = false
readingTime = false
+++

# Introduction

In the world of chatbots, natural language processing (NLP) has taken center stage, and Hugging Face Transformers has emerged as a powerful toolkit for working with NLP models. Slack, a popular team collaboration platform, offers a great environment for deploying custom bots to enhance communication within your workspace. In this blog post, I'll walk you through the process of creating a Slack bot powered by Hugging Face Transformers. By the end, you'll have a conversational bot capable of performing various NLP tasks right in your Slack channels.

# Prerequisites
Before we dive into the development process, let's make sure we have the necessary tools and accounts set up:

Slack Account: You need an active Slack account to create and deploy your bot within your workspace.

Dependencies: Install the dependencies required by cloning https://github.com/elliott-davis/slack_thread_summarizer and running `pip install -r ./requirements.txt`.

Bot Token: To create a Slack bot, you'll need to set up a Slack App and generate a bot token. This token will enable your bot to interact with your Slack workspace.

# Setting Up Your Slack Bot
Create a Slack App:

Go to the Slack API and click on "Create an App."
Give your app a name, select your workspace, and click "Create App."
Enable OAuth & Permissions:

In your app settings, navigate to "OAuth & Permissions."
Add the following Bot Token Scopes:
* app_mentions:read
* channels:history
* channels:join
* channels:read
* chat:write
* commands
* groups:history
* im:history
* mpim:history
* users:read
* commands
Save your changes.
Install Your App to Workspace:

In your app settings, go to "Install App" and click the "Install to Workspace" button.
Follow the prompts to grant permissions to your bot in your workspace.
Note the Bot Token:

In your app settings, go to "OAuth & Permissions" and copy the Bot User OAuth Token. You'll need this token to authenticate your bot.

# Building the Slack Bot
Now that you have your Slack bot set up, let's create a Python script that uses Hugging Face Transformers to power your bot.

```python
from haystack.nodes import TransformersSummarizer
from haystack import Document
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

summarizer = TransformersSummarizer(
    model_name_or_path="philschmid/flan-t5-base-samsum")

name_db = {}

# Cache the user ID's and real names from the thread so we don't blow past our rate limit
def get_real_name(client, id: str) -> str:
    if id in name_db:
        return name_db[id]
    real_name = client.users_info(user=id)["user"]["real_name"]
    name_db[id] = real_name
    return real_name

@app.event("app_mention")
def summarize_thread(client, event, say):
    
    thread_ts = event.get("thread_ts", None)
    if thread_ts is None:
        say("Sorry, I can only summarize threads")
        return
    
    channel_id = event.get("channel", None)
    result = client.conversations_replies(channel=channel_id, ts=thread_ts)
    
    if not result["ok"]:
        say("failed to get thread text. Try again later", thread_ts=thread_ts)
        return
    
    # Combine the content of the thread into a single document formatted like:
    # Elliott Davis: And he said whaaaaa
    # Matt Dresser: Nooooo
    conversation = Document('\n'.join(get_real_name(
        client, message["user"]) + ": " + message["text"] for message in result["messages"][:-1]))
    summary = summarizer.predict(documents=[conversation])

    say(text=summary[0].meta["summary"], thread_ts=thread_ts)

if __name__ == "__main__":
    # Used in dev mode since work blocks ngrok
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
```

In this script, I leverage the slack API to ensure that we are in a thread, then read all messages in that thread into a single string. This string is precisely formatted to match the models training data from the samsum dataset. After formatting the text, we can feed it into the transformer model to make a prediction on the summarized text and return it to slack.

# Conclusion
Creating a Slack bot powered by Hugging Face Transformers and haystack can greatly enhance communication and productivity in your workspace. With the capabilities of NLP at your bot's disposal, you can automate tasks, answer questions, and assist your team in various ways. By following the steps outlined in this blog post, you can bring your NLP-powered Slack bot to life and make your workspace even more dynamic and interactive.





