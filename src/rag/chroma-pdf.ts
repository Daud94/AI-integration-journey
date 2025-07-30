import {ChatOpenAI, OpenAIEmbeddings} from "@langchain/openai";
import {ChatPromptTemplate} from "@langchain/core/prompts";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import {PDFLoader} from "@langchain/community/document_loaders/fs/pdf";
import {MemoryVectorStore} from "langchain/vectorstores/memory";

const model = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0.7,
});

const embedding = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    openAIApiKey: process.env.OPENAI_API_KEY,
})

const question = "List the technical and soft skills this candidate has";

async function main() {
    try {
        // create the loader
        const loader = new PDFLoader('./resume.pdf', {splitPages: false})
        const docs = await loader.load()
        console.log(`Loaded ${docs.length} documents`)

        // Check if documents were loaded properly
        if (docs.length === 0) {
            throw new Error("No documents loaded from PDF");
        }

        // Debug: Check the content of loaded documents
        console.log('First document metadata:', docs[0].metadata)
        console.log('First document content length:', docs[0].pageContent.length)
        console.log('First document content preview:', docs[0].pageContent.substring(0, 200))

        // Check if the document actually has content
        if (!docs[0].pageContent || docs[0].pageContent.trim().length === 0) {
            throw new Error("PDF was loaded but contains no extractable text content. This might be a scanned PDF or image-based PDF.");
        }

        // split the docs with more lenient settings:
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 500,
            chunkOverlap: 50,
            separators: ['\n\n', '\n', '. ', ' ', '']
        })

        const splittedDocs = await splitter.splitDocuments(docs)
        console.log(`Split into ${splittedDocs.length} chunks`)

        // Debug: Show some chunk info
        if (splittedDocs.length > 0) {
            console.log('First chunk preview:', splittedDocs[0].pageContent.substring(0, 100))
        }

        // Check if we have valid chunks
        if (splittedDocs.length === 0) {
            console.log('Document content that failed to split:', docs[0].pageContent)
            throw new Error("No document chunks created - text splitter couldn't process the content");
        }

        // Use MemoryVectorStore instead of Chroma to avoid the embedding validation issue
        console.log('Creating vector store...')
        const vectorStore = await MemoryVectorStore.fromDocuments(
            splittedDocs,
            embedding
        )

        console.log('Vector store created successfully')

        // create data retriever
        const retriever = vectorStore.asRetriever({
            k: 2
        })

        console.log('Searching for relevant documents...')
        // get relevant documents
        const results = await retriever.invoke(question)
        console.log(`Found ${results.length} relevant documents`)

        const resultDocs = results.map(result => result.pageContent)

        // build template - fixed template structure
        const template = ChatPromptTemplate.fromMessages([
            ["system", "Answer the user's question based on the following context: {context}"],
            ["user", "{input}"],
        ])

        const chain = template.pipe(model)
        console.log('Generating response...')

        const response = await chain.invoke({
            input: question,
            context: resultDocs.join('\n\n')
        })

        console.log('\n--- RESPONSE ---')
        console.log(response.content)
    } catch (e: any) {
        console.error('Error occurred:', e.message)
        console.error('Full error:', e)
    }
}

main()