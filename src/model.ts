import {ChatOpenAI} from "@langchain/openai"

const model = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0.8,
    maxTokens: 700,
    // verbose: true,
})

async function main(){
    // const response = await model.invoke('Give me 4 good books to read')
    // console.log(response.conte nt)

    // const response = await model.batch([
    //     'Hello',
    //     'Give me 4 good books to read'
    // ])
    // console.log(response)

    const response = await model.stream('Give me 4 good books to read')
    for await (const chunk of response) {
        console.log(chunk.content)
    }
}

main()