import {ChatOpenAI} from "@langchain/openai";
import {ChatPromptTemplate} from "@langchain/core/prompts";
import {makeInvalidToolCall} from "@langchain/core/dist/output_parsers/openai_tools";

const model = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0.7,
});

async function fromTemplate() {
    const prompt = ChatPromptTemplate.fromTemplate(
        "Write a short description for the following product: {product_name}."
    );

    // const wholePrompt = await prompt.format({
    //     product_name: "iPhone 13 Pro Max",
    // })

    // creating a chain: connecting the model with the prompt
    const chain = prompt.pipe(model);
    const response = await chain.invoke({
        product_name: 'bicycle'
    })

    console.log(response.content)
}

async function fromMessage(){
    const prompt = ChatPromptTemplate.fromMessages([
        {role: "system", content: "Write a short description of the product provided by the user"},
        {role: "user", content: "{product_name}"},
    ]);

    const chain = prompt.pipe(model);
    const response = await chain.invoke({
        product_name: 'bicycle'
    })
    console.log(response.content)
}

fromMessage()