import {ChatOpenAI, OpenAIEmbeddings} from "@langchain/openai";
import {MemoryVectorStore} from "langchain/vectorstores/memory";
import {ChatPromptTemplate} from "@langchain/core/prompts";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import {PDFLoader} from "@langchain/community/document_loaders/fs/pdf";

const model = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0.7,
});

// const myData = [
//     "My name is John",
//     "My name is Jane",
//     "My favorite food is pizza",
//     "My favorite food is pasta",
// ]

const question = "List the technical and soft skills this candidate has";

async function main() {
    // create the loader
    const loader = new PDFLoader('resume.pdf', {splitPages: false})
    const docs = await loader.load()

    // split the docs:
    const splitter = new RecursiveCharacterTextSplitter({
        separators: [`. \n`]
    })

    const splittedDocs = await splitter.splitDocuments(docs)

    // store the data
    try {
        const vectorStore = new MemoryVectorStore(new OpenAIEmbeddings())
        await vectorStore.addDocuments(splittedDocs)

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