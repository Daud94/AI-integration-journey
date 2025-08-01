import {ChatOpenAI, OpenAIEmbeddings} from "@langchain/openai";
import {MemoryVectorStore} from "langchain/vectorstores/memory";
import {Document} from "@langchain/core/documents";
import {ChatPromptTemplate} from "@langchain/core/prompts";

const model = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0.7,
});

const myData = [
    "My name is John",
    "My name is Jane",
    "My favorite food is pizza",
    "My favorite food is pasta",
]

const question = "What are my favorite foods?";

async function main() {
    // store the data
    try {
        const vectorStore = new MemoryVectorStore(new OpenAIEmbeddings())
        await vectorStore.addDocuments(myData.map(content => new Document({pageContent: content})))

        // create data retriever
        const retriever = vectorStore.asRetriever({
            k: 2
        })

        // get relevant documents
        const results = await retriever.invoke(question)
        const resultDocs = results.map(result => result.pageContent)

        // build template
        const template = ChatPromptTemplate.fromMessages([
            {role: "system", content: "Answer the user's question based on the following context: {context}"},
            {role: "user", content: '{input}'},
        ])

        const chain = template.pipe(model)
        const response = await chain.invoke({
            input: question,
            context: resultDocs.join('\n')
        })
        console.log(response.content)
    } catch (e) {
        console.error(e)
    }

}

main()