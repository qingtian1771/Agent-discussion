# 对于Agent的一些定义

## LangChain中的定义
在LangChain的官方文档中对Agent是这么说的：
The core idea of agents is to use a language model to choose a sequence of actions to take. In chains, a sequence of actions is hardcoded (in code). In agents, a language model is used as a reasoning engine to determine which actions to take and in which order.

Agent的核心思想是使用语言模型来选择要采取的一系列动作（Action）。 在Agent中，语言模型被用作推理引擎来确定要采取哪些动作以及按什么顺序。所以在langChain的定义中，Agent的作用主要是选择动作。

## crewAI中的定义
crewAI对Agent是这么定义的：
An agent is an autonomous unit programmed to:

Perform tasks
Make decisions
Communicate with other agents

Think of an agent as a member of a team, with specific skills and a particular job to do. Agents can have different roles like 'Researcher', 'Writer', or 'Customer Support', each contributing to the overall goal of the crew.

Agent是一个自动化的单元，主要是用于：

完成任务。
做出决策。
和其它Agent去交流。

可以把Agent想象成是一个队伍中的成员，每一个成员都有其特殊的能力和工作，也就是在一个队伍中承担不同的角色，它们之间通过合作去完成一个任务。


# LangChain中Agent的相关概念

## AgentAction
This is a dataclass that represents the action an agent should take. It has a tool property (which is the name of the tool that should be invoked) and a tool_input property (the input to that tool)

它是一个数据类，用于表示agent要采取的动作，它是一个包含两个属性的类，这两个属性是“tool”和“tool_input”，分表表示tool的名称和输入。

## AgentFinish
This represents the final result from an agent, when it is ready to return to the user. It contains a return_values key-value mapping, which contains the final agent output. Usually, this contains an output key containing a string that is the agent's response.

表示Agent的最终结果，是一个kay-value的mapping,包含agent的最终输出结果。

## Intermediate Steps
These represent previous agent actions and corresponding outputs from this CURRENT agent run. These are important to pass to future iteration so the agent knows what work it has already done. This is typed as a List[Tuple[AgentAction, Any]]. Note that observation is currently left as type Any to be maximally flexible. In practice, this is often a string.

它表示当前agent运行时，之前所有的Action和action相应的输出。

## LangChain中Agent的表示
This is the chain responsible for deciding what step to take next. This is usually powered by a language model, a prompt, and an output parser. Different agents have different prompting styles for reasoning, different ways of encoding inputs, and different ways of parsing the output.

这一段话有下面几个含义：
Agent是一个chain，用来负责决定下一步采取什么步骤。
这个chain，通常由由语言模型、提示和输出解析器构成，也就是我们常见的不同的chain。
不同的智能体有不同的推理提示风格、不同的编码输入方式以及不同的解析输出方式。

## Agent Inputs
The inputs to an agent are a key-value mapping. There is only one required key: intermediate_steps, which corresponds to Intermediate Steps as described above.
Generally, the PromptTemplate takes care of transforming these pairs into a format that can best be passed into the LLM.

Agent的输入是key-value的mapping，也就是上面提到过的intermediate_steps，PromptTemplate负责将这些mapping转换成语言模型可以输入的格式。


## Agent Outputs
The output is the next action(s) to take or the final response to send to the user (AgentActions or AgentFinish). Concretely, this can be typed as Union[AgentAction, List[AgentAction], AgentFinish].

The output parser is responsible for taking the raw LLM output and transforming it into one of these three types.

Agent的输出是下一个（或多个）Action，或者是最终的结果，具体它被定义成Union[AgentAction, List[AgentAction], AgentFinish]。output parser负责将原始的LLM输出变换成Union中的三个具体类型之一。


## AgentExecutor
The agent executor is the runtime for an agent. This is what actually calls the agent, executes the actions it chooses, passes the action outputs back to the agent, and repeats. In pseudocode, this looks roughly like:

    next_action = agent.get_action(...)  //获取下一个action
    while next_action != AgentFinish:   //如果下一个action不是AgentFinish
      observation = run(next_action)    //那么运行下一个action
      next_action = agent.get_action(..., next_action, observation) //根据action和它的结果，再获取下一个action
    return next_action  //返回下一个action

While this may seem simple, there are several complexities this runtime handles for you, including:

1. Handling cases where the agent selects a non-existent tool
2. Handling cases where the tool errors
3. Handling cases where the agent produces output that cannot be parsed into a tool invocation
4. Logging and observability at all levels (agent decisions, tool calls) to stdout and/or to LangSmith.

Agent executor负责运行一个agent，从上面的伪代码可以看出它做的事情：主要是通过让agent获取下一个action，并且运行这个action，然后再获取下一个action，直到这个action是AgentFinish为止。

Agent executor还负责处理意外，还有做一些logging的事情：

1. 处理Agent选择不存在tool的情况。
2. tool出错的情况处理。
3. 处理Agent生成的输出无法解析为tool调用需要的输入的情况。
4. 所有级别（agent决策、tool调用）到标准输出，和/或 LangSmith 的日志记录和可观察性。


## Tools
Tools are functions that an agent can invoke. The Tool abstraction consists of two components:

- The input schema for the tool. This tells the LLM what parameters are needed to call the tool. Without this, it will not know what the correct inputs are. These parameters should be sensibly named and described.
- The function to run. This is generally just a Python function that is invoked.

### Considerations
There are two important design considerations around tools:

1. Giving the agent access to the right tools
2. Describing the tools in a way that is most helpful to the agent

Without thinking through both, you won't be able to build a working agent. If you don't give the agent access to a correct set of tools, it will never be able to accomplish the objectives you give it. If you don't describe the tools well, the agent won't know how to use them properly.

LangChain provides a wide set of built-in tools, but also makes it easy to define your own (including custom descriptions). For a full list of built-in tools, see the [tools integrations section](https://python.langchain.com/docs/integrations/tools/).

tool是agent可以调用的函数。tool需要知道两个事情：一个是函数的输入格式，知道这个的目的是让LLM了解应该给tool准备什么样的输入数据，所以参数的名称和描述都应该是有意义的；另一个就是需要调用的函数，通常是一个python函数。

所以设计的时候需要考虑下面两点：
1. 为agent提供正确的工具。
2. 用对agent最有帮助的方式描述工具。

如果不考虑这两点，你将无法构建一个有效的agent。 如果您不让agent访问一组正确的工具，它将永远无法实现您赋予它的目标。 如果你没有很好地描述工具，agent将不知道如何正确使用它们。


## Toolkits
For many common tasks, an agent will need a set of related tools. For this LangChain provides the concept of toolkits - groups of around 3-5 tools needed to accomplish specific objectives. For example, the GitHub toolkit has a tool for searching through GitHub issues, a tool for reading a file, a tool for commenting, etc.

LangChain provides a wide set of toolkits to get started. For a full list of built-in toolkits, see the [toolkits integrations section](https://python.langchain.com/docs/integrations/toolkits/)

一个tool对应一个函数，一个toolkits就是一组相关的函数（tool），例如Github toolkit就是由许多tool组成：用于搜索GitHub问题的tool、用于读取文件的tool、用于评论的tool等等。


## Agent Types
This categorizes all the available agents along a few dimensions.
Agent有一些不同的类型，区分agent的类型，主要是从下面的这几个维度。

- Intended Model Type

Whether this agent is intended for Chat Models (takes in messages, outputs message) or LLMs (takes in string, outputs string). The main thing this affects is the prompting strategy used. You can use an agent with a different type of model than it is intended for, but it likely won't produce results of the same quality.

Agent有它偏好的模型，主要是对话模型（接收消息，输出消息）和一般LLM（接收字符串，输出字符串）的区分。这影响的主要因素是所使用的提示策略。 您可以使用具有与预期不同类型模型的代理，但它可能不会产生相同质量的结果。


- Supports Chat History

Whether or not these agent types support chat history. If it does, that means it can be used as a chatbot. If it does not, then that means it's more suited for single tasks. Supporting chat history generally requires better models, so earlier agent types aimed at worse models may not support it.

- Supports Multi-Input Tools

Whether or not these agent types support tools with multiple inputs. If a tool only requires a single input, it is generally easier for an LLM to know how to invoke it. Therefore, several earlier agent types aimed at worse models may not support them.

- Supports Parallel Function Calling

Having an LLM call multiple tools at the same time can greatly speed up agents whether there are tasks that are assisted by doing so. However, it is much more challenging for LLMs to do this, so some agent types do not support this.

- Required Model Params

Whether this agent requires the model to support any additional parameters. Some agent types take advantage of things like OpenAI function calling, which require other model parameters. If none are required, then that means that everything is done via prompting

- When to Use

Our commentary on when you should consider using this agent type.

LangChain中有文档专门讨论[agent的类型](https://python.langchain.com/docs/modules/agents/agent_types/)，详细的可以参考这个内容。在这个文档中，我认为仅仅需要注意“Tool Calling”这个类型的agent就可以，因为这个类型是最具代表性的，其它的一些旧的类型，例如：“OpenAI Tools”和“OpenAI Functions”都会被这个类型给替代了。


