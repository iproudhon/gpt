#!/bin/bash

':' //; export NODE_OPTIONS=--experimental-repl-await;
':' //; export NODE_PATH=$(npm root -gq):$(npm root -q):.
':' //; [ $# -le 0 ] && exec "$(which node)" -r $0 || exec "$(which node)" -r $0 $0 $*

// gpt.js
//
// npm install -g openai

const openai = require("openai")

loadFile = function(fn) {
  return require('fs').readFileSync(fn).toString()
}

saveFile = function(fn, data) {
  return require('fs').writeFileSync(fn, data)
}

loadScript = function(fn) {
  // eval in global scope
  eval.apply(this, [ loadFile(fn) ])
  // eval.apply(global, [ loadFile(fn) ])
}

class GPT {
  constructor(model) {
    // gpt-4, gpt-4-32k
    // gpt-4-1106-preview, a.k.a. gpt-4-turbo
    // gpt-4-1106-vision-preview
    this.model = model || "gpt-4-1106-preview" || "gpt-3.5-turbo"
    this.messages = [
      { role: "system", content: "You're an expert coder and a sharp critic." },
    ]
    this.initializeOpenAIApi()
  }

  async initializeOpenAIApi() {
//    const openai = await import("openai")
    this.chat = new openai.OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    })
  }

  // Make sure the function is defined as a class method and uses async/await
  async send(msg, stream) {
    this.messages.push({ role: "user", content: msg })
    if (!stream) {
      const reply = await this.chat.chat.completions.create({
        model: this.model,
        messages: this.messages,
      })
      const responseMessage = reply.choices[0].message
      this.messages.push(responseMessage)
      return responseMessage.content
    } else {
      const reply = await this.chat.chat.completions.create({
        model: this.model,
        messages: this.messages,
        stream: true,
      })
      var msg = { role: "assistant", content: "" }
      for await (const r of reply) {
        var delta = r.choices[0].delta.content
        if (delta) {
          msg.content += delta
          process.stdout.write(delta)
        }
      }
      this.messages.push(msg)
      process.stdout.write('\n')
      return msg.content
    }
  }
}

global.GPT = GPT
global.gpt = new GPT()

// EOF
