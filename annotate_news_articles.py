import pickle
import pandas as pd
import pyperclip



with open("artifacts/annotated_data_dict.pkl", "rb") as f:
    annotated_data: dict = pickle.load(f)

text = '''Meta to pay nearly $15 billion for Scale AI stake, The Information reports. June 10 (Reuters) - Meta Platforms (META.O), opens new tab has agreed to take a 49% stake in artificial intelligence startup Scale AI for $14.8 billion, The Information reported on Tuesday, citing two people familiar with the matter.
Founded in 2016, Scale AI provides vast amounts of labeled data or curated training data, which is crucial for developing sophisticated tools such as OpenAI's ChatGPT.The deal, which has not been finalized yet, appears to be beneficial for Scale AI's investors including Accel, Index Ventures, Founders Fund and Greenoaks, as well as its current and former employees, the report said.
Meta, Scale AI and the startup's investors did not immediately respond to Reuters' requests for comment.
As part of the deal, Scale AI CEO Alexandr Wang will take a top position inside Meta, leading a new "superintelligence" lab, according to the report.
Meta CEO Mark Zuckerberg has been actively recruiting top AI researchers to boost the company's AI efforts, the report said.
The company is fighting the perception that it may have fallen behind in the AI race after its initial set of Llama 4 large language models released in April fell short of performance expectations.
Meta delayed the release of its flagship "Behemoth" AI model due to concerns about its capabilities, the Wall Street Journal reported last month.
The company is also facing antitrust concerns related to its acquisitions of Instagram and WhatsApp.
According to The Information report, the structure for the potential deal with Scale AI could be designed to avoid more regulatory scrutiny.
Scale AI was valued at $13.8 billion in a funding round last spring. It generated about $870 million in revenue in 2024 and expects more than $2 billion this year, the report said.
The company had more than $900 million of cash on its balance sheet at the end of last year, according to the report.'''
pyperclip.copy(text)
annotation = [
  {
    "label": ["Company Buying Startup"],
    "points": [
      {
        "start": 0,
        "end": 4,
        "text": "Meta"
      },
      {
        "start": 105,
        "end": 119,
        "text": "Meta Platforms"
      },
      {
        "start": 636,
        "end": 640,
        "text": "Meta"
      },
      {
        "start": 806,
        "end": 810,
        "text": "Meta"
      },
      {
        "start": 866,
        "end": 870,
        "text": "Meta"
      }
    ]
  },
  {
    "label": ["Funding Amount"],
    "points": [
      {
        "start": 20,
        "end": 31,
        "text": "$15 billion"
      },
      {
        "start": 197,
        "end": 210,
        "text": "$14.8 billion"
      }
    ]
  },
  {
    "label": ["Startup Name"],
    "points": [
      {
        "start": 37,
        "end": 45,
        "text": "Scale AI"
      },
      {
        "start": 184,
        "end": 192,
        "text": "Scale AI"
      },
      {
        "start": 284,
        "end": 292,
        "text": "Scale AI"
      },
      {
        "start": 497,
        "end": 505,
        "text": "Scale AI"
      },
      {
        "start": 642,
        "end": 650,
        "text": "Scale AI"
      },
      {
        "start": 742,
        "end": 750,
        "text": "Scale AI"
      },
      {
        "start": 1335,
        "end": 1343,
        "text": "Scale AI"
      },
      {
        "start": 1392,
        "end": 1400,
        "text": "Scale AI"
      }
    ]
  },
  {
    "label": ["Investment Company"],
    "points": [
      {
        "start": 529,
        "end": 534,
        "text": "Accel"
      },
      {
        "start": 536,
        "end": 550,
        "text": "Index Ventures"
      },
      {
        "start": 552,
        "end": 565,
        "text": "Founders Fund"
      },
      {
        "start": 570,
        "end": 579,
        "text": "Greenoaks"
      }
    ]
  },
  {
    "label": ["Founder Name"],
    "points": [
      {
        "start": 755,
        "end": 768,
        "text": "Alexandr Wang"
      }
    ]
  },
  {
    "label": ["Valuation"],
    "points": [
      {
        "start": 1415,
        "end": 1428,
        "text": "$13.8 billion"
      }
    ]
  },
  {
    "label": ["Revenue"],
    "points": [
      {
        "start": 1465,
        "end": 1478,
        "text": "$870 million"
      },
      {
        "start": 1522,
        "end": 1532,
        "text": "$2 billion"
      }
    ]
  }
]

annotated_data["news_articles"].append(text)
annotated_data["annotations"].append(annotation)

# annotated_data["annotations"][9] = 

with open("artifacts/annotated_data_dict.pkl", "wb") as f:
    pickle.dump(annotated_data, f)


        

