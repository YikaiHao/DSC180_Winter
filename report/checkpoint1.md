# DSC180A Checkpoint 1 - A04 Group 2
Member: Ruoyu Liu, Yu-Chieh Chen, Yikai Hao

## Introduction
With more people starting to use cellphones and store their information on cellphones, it is important to protect user’s information. The Android system is the most widely used mobile operation system which takes part of 80% of the market. Since the Android environment is open source, everyone is able to upload their apps and let users try on their apps. This leads to a severe problem of malwares on the Android system. Malwares are split into multiple categories - trojans, adware, etc. Each type of malware has different performance, they might send out messages unconsciously, steal the user's information, or lock down the phone until the user pays the ransom.

With the development of the market, there already exist lots of methods to detect malwares. However, the majority of those methods focus on the behaviour that malware has. With the development of technology, people start to think about using the machine learning techniques on malware detection. In order to detect the malware at the born stage, we try a new method which tries to analyse the code of apps and detect malware by checking the similarities among their code, including the apis they used and the relationships between apis. If we are able to detect the malware at the born stage, we can future collaborate with third party app stores to detect the apps before they are published. In this way, we will be able to benefit millions of users. 

## Data 
In order to detect malware, we need to understand the code from Android apps. Android apps are written in Java, compiled to Dalvik bytecode, and distributed in apk (Android Application Package) files. In order to analyze the code, we decompile these Android Apk files (.dex) into readable Smali code using ApkTool. 

### Smali File 
To analyze code in the Smali file, we need to understand the structure of it. The following code is a sample Smali file from Microsoft Word app. 

```
.class final Lbolts/a;
.super Ljava/lang/Object;
.source "SourceFile"

# static fields
.field static final a:I
.field static final b:I

# direct methods
.method static constructor <clinit>()V
    ...
    .line 58
    invoke-static {}, Ljava/lang/Runtime;->getRuntime()Ljava/lang/Runtime;
    ...
    return-void
.end method
```

The basic components of a Smali file are the following: 
1. Class information: In this example, Lbolts/a is the class name for this file. 
2. Static fields 
3. Method direct virtual

### API Call 
To understand the key features of malware, we focus on the API (Application Programming Interface) calls in the Smali files. The following is an example of the API call which has four main components. 

```invoke-static {}, Ljava/lang/Runtime;->getRuntime()Ljava/lang/Runtime;```

The basic components of an API call are the following: 
1. Invoke method: there are five methods to invoke in API calls, including invoke-static, invoke-virtual, invoke-direct, invoke-super, and invoke-interface.
2. API package 
3. Method name  
4. Return type of the method

### Dataset 
The dataset that is being investigated includes three types of apps, including malware, random, and popular apps. There are 4963 apps in malware-apps, 234 apps in random-apps, and 325 apps in popular-apps. In the malware apps, there are 22 different malware types.  

### Feature Extraction
 Our approach is to use os.walk to read through all the Smali files in different apps. For the baseline model, we extract features including the number of unique api, blocks, and files by using regex. For our current model, we extract features, such as package names, invoke type, api name, code block number, app name, class name, and type of apps. These data can help us to determine the relationship between different apps. Also, it helps us to construct the A, B, and P matrix to understand the suspicious api calls to be malware. 
 
Since our detection method is based on the apis and the relationship between apis, the goal of the data we extract out from the original raw data will be fully represented apis and relationships among them. Therefore, the invoke method, api name, the method, and the return type fully represent most of the features in an API. In addition, we also saved the names of the APIs, the block numbers, which shows whether the APIs are in the same code block, and the types of the app. Therefore, we can find some relationships among APIs from those data. For example, B matrix shows APIs that appear in the same block of code. Besides this, some apps might use unique APIs that appeared differently from popular apps and random apps. By using the type of the app as the label in future models, we will be able to find the relationships between the apps, and check the similarities between apps. This will help us find the malware. Therefore, the data we get is enough to represent the app and appropriate for future work.

