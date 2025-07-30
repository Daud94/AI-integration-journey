import {ChatOpenAI} from "@langchain/openai";
import {ChatPromptTemplate} from "@langchain/core/prompts";
import {StringOutputParser, CommaSeparatedListOutputParser, StructuredOutputParser} from "@langchain/core/output_parsers";

const model = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0.7,
});

async function stringParser() {
    const prompt = ChatPromptTemplate.fromTemplate(
        "Write a short description for the following product: {product_name}."
    );

    const parser = new StringOutputParser();


    // creating a chain: connecting the model with the prompt
    const chain = prompt.pipe(model).pipe(parser);
    const response = await chain.invoke({
        product_name: 'bicycle'
    })
    console.log(response)
}

async function commaSeparatedListParser() {
    const prompt = ChatPromptTemplate.fromTemplate(
        "Provide the first 5 ingredients, seperated by comma for: {word}."
    );

    const parser = new CommaSeparatedListOutputParser();


    // creating a chain: connecting the model with the prompt
    const chain = prompt.pipe(model).pipe(parser);
    const response = await chain.invoke({
        word: 'bicycle'
    })
    console.log(response)
}

async function structuredParser(){
    const templatePrompt = ChatPromptTemplate.fromTemplate(`
    Extract information from the following phrase.
    Formatting instructions: {format_instructions}
    Phrase: {phrase}`)

    const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
        name: 'the name of the person',
        likes: 'What the person likes'
    })

    const chain = templatePrompt.pipe(model).pipe(outputParser)
    const response = await chain.invoke({
        phrase: 'I like to play video games',
        format_instructions: outputParser.getFormatInstructions()
    })
    console.log(response)
}



structuredParser()