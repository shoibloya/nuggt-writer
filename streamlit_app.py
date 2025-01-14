import streamlit as st
import os
import json
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------------------------
# 1) Setup
# ------------------------------------------------------------------------------
# It's recommended to store API keys securely using Streamlit's secrets management.
# Ensure you have a `.streamlit/secrets.toml` file with the following structure:
#
# [secrets]
# TAVILY_API_KEY = "your_tavily_api_key"
# OPENAI_API_KEY = "your_openai_api_key"

# Ensure environment variables are properly set
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = st.secrets['tapiKey']

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets['apiKey']

# Instantiating Tavily client
tavily_client = TavilyClient()

# LLM Configuration
llm = ChatOpenAI(model="gpt-4o", temperature=0)
str_parser = StrOutputParser()

st.set_page_config(page_title="Advanced Blog Generator", layout="wide")

# Initialize session state variables
if "outlines_proposed" not in st.session_state:
    st.session_state["outlines_proposed"] = []
if "blogs_generated" not in st.session_state:
    st.session_state["blogs_generated"] = []
if "all_research" not in st.session_state:
    st.session_state["all_research"] = ""
if "company_info" not in st.session_state:
    st.session_state["company_info"] = ""
if "product_info" not in st.session_state:
    st.session_state["product_info"] = ""
if "current_step" not in st.session_state:
    st.session_state["current_step"] = "input"
if "final_blog" not in st.session_state:
    st.session_state["final_blog"] = ""

# ------------------------------------------------------------------------------
# 2) Utility Functions with Enhanced Prompts
# ------------------------------------------------------------------------------
def present_extracted_info(raw_content, subject="the company/product"):
    """
    Summarizes and presents extracted raw content in a clear, concise manner.
    """
    sys_prompt = f"""
    You are an expert content analyst. The user has extracted raw content about {subject} from a website. 
    Your task is to summarize and present this information clearly and concisely. 
    Focus on key aspects such as mission, products/services, unique selling points, and any interesting data or facts.
    Ensure the summary is well-structured, easy to read, and free of unnecessary jargon.
    
    GOOD OUTPUT:
    - Clear headings and subheadings
    - Concise bullet points highlighting key information
    - Proper grammar and formatting
    
    BAD OUTPUT:
    - Vague statements without supporting details
    - Overly long paragraphs without structure
    - Inclusion of irrelevant information
    """

    chat_prompt = ChatPromptTemplate(
        input_variables=["extracted"],
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=[],
                    template=sys_prompt
                )
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["extracted"],
                    template='Extracted Content:\n"""{extracted}"""'
                )
            ),
        ]
    )
    # Render prompt
    prompt_msgs = chat_prompt.format_prompt(extracted=raw_content).to_messages()
    response = llm.invoke(prompt_msgs)
    return response.content


def summarize_tavily_search(results):
    """
    Summarizes raw search results into bullet points with source URLs.
    """
    sys_prompt = """
    You are an expert researcher. You have been provided with raw search results from Tavily. 
    Your task is to extract key data, insights, and statistics, presenting them in clear bullet points.
    Each bullet should include relevant information and, if available, the source URL as an inline citation.
    
    GOOD OUTPUT:
    - Bullet points with concise information
    - Inline citations linking to source URLs
    - Organized and easy to follow
    
    BAD OUTPUT:
    - Long, unstructured paragraphs
    - Missing or incorrect citations
    - Irrelevant or redundant information
    """

    raw_content = json.dumps(results, ensure_ascii=False)
    chat_prompt = ChatPromptTemplate(
        input_variables=["search_results"],
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=[],
                    template=sys_prompt
                )
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["search_results"],
                    template='Information: """{search_results}"""'
                )
            )
        ]
    )
    msgs = chat_prompt.format_prompt(search_results=raw_content).to_messages()
    response = llm.invoke(msgs)
    return response.content


def generate_product_queries(product_name, company_desc):
    """
    Generates 3 sophisticated search queries about the product, incorporating company description.
    """
    sys_prompt = f"""
    You are an expert market analyst. The user has a product named '{product_name}' from a company described as follows:
    "{company_desc}"
    
    Your task is to generate exactly 3 advanced search queries that delve deeper into understanding the product.
    These queries should explore areas such as features, user reviews, competitor comparisons, and other relevant angles.
    
    Each query should be presented as a bullet point.
    
    GOOD OUTPUT:
    - Well-thought-out queries targeting specific aspects of the product
    - Concise and free of unnecessary wording
    
    BAD OUTPUT:
    - Generic or vague queries without clear focus
    - Overly verbose or confusing language
    """

    msgs = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content="Please provide 3 advanced queries.")
    ]
    resp = llm.invoke(msgs)
    return resp.content


def generate_5_queries(blog_topic, mandatory_items):
    """
    Generates 5 research queries based on blog topic and mandatory items.
    """
    if mandatory_items.strip():
        prompt_mandatory = f"""
        You are an expert content strategist. The user has a blog topic: '{blog_topic}' and has specified the following mandatory items: '{mandatory_items}'.
        
        Your task is to generate exactly 5 comprehensive search queries that focus on these mandatory items while remaining relevant to the blog topic.
        Each query should be a bullet point, aiming to uncover in-depth information and insights.
        
        GOOD OUTPUT:
        - Detailed and specific queries targeting mandatory items
        - Relevant to the overall blog topic
        - Clear and concise
        - your queries are simple google searches. do not write sophisticated google searches. each search should focus on one thing.
        
        BAD OUTPUT:
        - Overly broad or irrelevant queries
        - Lack of focus on mandatory items
        - Unclear or poorly structured queries
        """
    else:
        prompt_mandatory = f"""
        You are an expert content strategist. The user has a blog topic: '{blog_topic}'.
        
        Your task is to generate exactly 5 comprehensive search queries that are highly relevant to this topic.
        Each query should be a bullet point, designed to uncover in-depth information and valuable insights.
        
        GOOD OUTPUT:
        - Detailed and specific queries relevant to the blog topic
        - Clear and concise
        
        BAD OUTPUT:
        - Overly broad or irrelevant queries
        - Vague or poorly structured queries
        """

    msgs = [
        SystemMessage(content=prompt_mandatory),
        HumanMessage(content="Please provide 5 comprehensive queries.")
    ]
    resp = llm.invoke(msgs)
    return resp.content


def generate_outline(
    blog_topic, target_audience, key_focus, article_length,
    style_tone, keywords, cta_text, additional_blog_insights,
    company_desc, product_desc, mandatory_summaries,
    previous_outlines  # New parameter to receive previous outlines
):
    """
    Generates a single sophisticated blog outline incorporating research and ensuring uniqueness.
    """
    # Combine previous outlines into a single string for context
    if previous_outlines:
        previous_outlines_text = "\n".join([f"Outline {idx+1}: {outline}" for idx, outline in enumerate(previous_outlines)])
        uniqueness_instruction = f"\n\nEnsure that this outline is significantly different from the following outlines:\n{previous_outlines_text}"
    else:
        uniqueness_instruction = ""

    sys_prompt = f"""
    You are an expert blog strategist. The user intends to write a blog with the following details:
    - **Topic:** {blog_topic}
    - **Target Audience:** {target_audience}
    - **Key Focus/Angle:** {key_focus}
    - **Desired Article Length:** {article_length}
    - **Style/Tone:** {style_tone}
    - **Keywords:** {keywords}
    - **Call to Action (CTA):** {cta_text}
    - **Additional Insights:** {additional_blog_insights}
    
    **Company Description:** {company_desc}
    **Product Description:** {product_desc}
    
    **Research Summaries:**
    {mandatory_summaries}
    
    **Task:**
    1. Develop a distinct blog outline with a unique structure and perspective.
    2. Provide approximate word distribution across sections.
    3. Incorporate key findings from the research, ensuring each section references relevant data or insights with source URLs.
    4. Present the outline with clear headings like "Outline 1", followed by subheadings for each section.
    
    **GUIDELINES FOR GOOD OUTPUT:**
    - The outline should offer a unique approach to the topic.
    - Logical flow with well-defined sections.
    - Integration of research with proper citations.
    - Clarity and conciseness in structuring the outline.
    
    **GUIDELINES FOR BAD OUTPUT:**
    - Repetitive or similar outlines without distinct differences.
    - Lack of structure or logical flow.
    - Missing or improper citations for research data.
    - Overly vague or generic sections without depth.
    {uniqueness_instruction}
    """

    chat_prompt = ChatPromptTemplate(
        input_variables=[],
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=[],
                    template=sys_prompt
                )
            ),
            HumanMessage(content="Please generate a distinct blog outline based on the above information.")
        ]
    )
    resp = llm.invoke(chat_prompt.format_prompt().to_messages())
    return resp.content



def generate_blog(
    chosen_outline, blog_topic, target_audience, key_focus, article_length,
    style_tone, keywords, cta_text, additional_blog_insights,
    company_desc, product_desc, mandatory_summaries
):
    """
    Generates a blog post based on the chosen outline, incorporating research with inline citations.
    """
    sys_prompt = f"""
    You are an expert content writer specializing in creating high-quality blog posts. Below are the details and the chosen outline for the blog:
    
    - **Topic:** {blog_topic}
    - **Target Audience:** {target_audience}
    - **Key Focus/Angle:** {key_focus}
    - **Desired Article Length:** {article_length}
    - **Style/Tone:** {style_tone}
    - **Keywords:** {keywords}
    - **Call to Action (CTA):** {cta_text}
    - **Additional Insights:** {additional_blog_insights}
    
    **Company Description:** {company_desc}
    **Product Description:** {product_desc}
    
    **Research Summaries:**
    {mandatory_summaries}
    
    **Chosen Outline:**
    {chosen_outline}
    
    **Task:**
    1. Write a comprehensive blog post following the chosen outline.
    2. Ensure each section adheres to the outline's structure and word distribution.
    3. Incorporate relevant data and insights from the research, citing sources with inline Markdown links (e.g., [Source](URL)).
    4. Maintain a natural, engaging tone suitable for the target audience.
    5. Ensure factual accuracy and coherence throughout the article.
    6. Conclude with a strong CTA if provided.
    
    **GUIDELINES FOR GOOD OUTPUT:**
    - Adheres strictly to the chosen outline and structure.
    - Seamlessly integrates research with proper inline citations.
    - Engaging and readable Markdown formatting.
    - Clear and persuasive CTA (if provided).
    
    **GUIDELINES FOR BAD OUTPUT:**
    - Deviates from the outline or structure.
    - Missing or improper citations for referenced data.
    - Poor readability or unengaging tone.
    - Incomplete sections or lack of depth.
    """

    msgs = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content="Please write the full blog article in Markdown format based on the above information.")
    ]
    resp = llm.invoke(msgs)
    return resp.content

def main():
    st.title("Advanced Blog Generator")

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Blog Inputs")
        blog_topic = st.text_input("Blog Topic", value="")
        target_audience = st.text_input("Target Audience", value="")
        key_focus = st.text_area("Key Focus / Angle", value="")
        article_length = st.selectbox(
            "Desired Article Length",
            ["Long (1200+ words)"]
        )
        style_tone = st.selectbox(
            "Style / Tone",
            ["Conversational", "Formal", "Data-heavy / Analytical", "Story-driven"]
        )
        keywords = st.text_area("Keywords (optional)", value="")
        cta_text = st.text_input("Preferred CTA (optional)", value="")
        additional_blog_insights = st.text_area("Additional Insights (optional)", value="")

        st.markdown("---")
        company_desc = st.text_area("Company Description (for context)", value="")
        company_url = st.text_input("Company Website URL (optional)", value="")
        product_name = st.text_input("Product Name (optional)", value="")
        product_url = st.text_input("Product URL (optional, content will be extracted)", value="")

        st.markdown("---")
        mandatory_items = st.text_area(
            "Mandatory Research (comma-separated, optional). e.g., 'latest AI usage stats, consumer trust data'",
            value=""
        )

        generate_flow = st.button("Start Content Generation Flow", key="generate_flow")

    if generate_flow:
        # Reset session state for a fresh start
        st.session_state["outlines_proposed"] = []
        st.session_state["blogs_generated"] = []
        st.session_state["all_research"] = ""
        st.session_state["current_step"] = "company_info"

        # ------------------------------------------------------------------------------
        # Step 1: Company Information Extraction
        # ------------------------------------------------------------------------------
        st.header("1) Company Information")
        if company_url.strip():
            st.info("Extracting content from company website...")
            try:
                with st.spinner("Extracting company information..."):
                    response = tavily_client.extract(urls=[company_url])
                    extracted_text = ""
                    for result in response["results"]:
                        extracted_text += f"URL: {result['url']}\nRaw Content:\n{result['raw_content']}\n\n"
                    # Summarize & present
                    summarized_company = present_extracted_info(extracted_text, subject="the company")
                    st.success("Company Info Summary:")
                    st.write(summarized_company)
                    st.session_state["company_info"] = summarized_company
            except Exception as e:
                st.error(f"Error extracting company info: {str(e)}")
                st.session_state["company_info"] = company_desc  # Fallback to user-provided description
        else:
            if company_desc.strip():
                st.warning("No company URL provided, using the provided company description.")
                st.session_state["company_info"] = company_desc
            else:
                st.warning("No company information provided.")
                st.session_state["company_info"] = ""

        # ------------------------------------------------------------------------------
        # Step 2: Product Information Extraction & Queries
        # ------------------------------------------------------------------------------
        st.header("2) Product Information")
        if product_url.strip():
            st.info("Extracting content from product page...")
            try:
                with st.spinner("Extracting product information..."):
                    response = tavily_client.extract(urls=[product_url])
                    extracted_text = ""
                    for result in response["results"]:
                        extracted_text += f"URL: {result['url']}\nRaw Content:\n{result['raw_content']}\n\n"
                    summarized_product = present_extracted_info(extracted_text, subject="the product")
                    st.success("Product Info Summary:")
                    st.write(summarized_product)
                    st.session_state["product_info"] = summarized_product
            except Exception as e:
                st.error(f"Error extracting product info: {str(e)}")
                st.session_state["product_info"] = product_name
        else:
            if product_name.strip():
                st.warning("Product URL not provided, using the product name/description.")
                st.session_state["product_info"] = product_name
            else:
                st.warning("No product information provided.")
                st.session_state["product_info"] = ""

        if product_name.strip() or product_url.strip():
            st.markdown("### Generating 3 Advanced Queries for the Product")
            try:
                with st.spinner("Generating product-specific queries..."):
                    product_queries_response = generate_product_queries(
                        product_name,
                        st.session_state["company_info"]
                    )
                    st.info("Product-Specific Queries:")
                    st.write(product_queries_response)

                    # Parse queries
                    lines = product_queries_response.strip().split("\n")
                    product_research_summary = ""
                    count_idx = 1
                    st.markdown("#### Product Queries: Searching & Summarizing")
                    for line in lines:
                        line = line.strip("-•* ")  # Remove bullet characters
                        if not line:
                            continue
                        # Treat line as the query
                        q_expander = st.expander(f"Query #{count_idx}: {line}")
                        with q_expander:
                            try:
                                sr = tavily_client.search(query=line, search_depth="advanced", include_raw_content=True)
                                results = sr["results"]
                                sum_text = summarize_tavily_search(results)
                                st.write(sum_text)
                                product_research_summary += f"\n\n**Query {count_idx}:** {line}\n{sum_text}"
                            except Exception as e:
                                st.error(f"Error occurred during search: {str(e)}")
                        count_idx += 1
                    st.session_state["all_research"] += product_research_summary
            except Exception as e:
                st.error(f"Error generating product queries: {str(e)}")

        # ------------------------------------------------------------------------------
        # Step 3: Mandatory/Topic-Based Research Queries
        # ------------------------------------------------------------------------------
        st.header("3) Mandatory / Topic-Based Research")
        try:
            with st.spinner("Generating mandatory/topic-based queries..."):
                five_queries_text = generate_5_queries(blog_topic, mandatory_items)
                st.info("Mandatory/Topic-Based Queries:")
                st.write(five_queries_text)

                # Parse queries
                lines_5 = five_queries_text.strip().split("\n")
                mandatory_summaries = ""
                query_counter = 1

                st.markdown("#### Research Queries: Searching & Summarizing")
                for l in lines_5:
                    l = l.strip("-•* ")
                    if not l:
                        continue
                    # Treat the entire line as the query
                    expander_label = f"Research Query #{query_counter}: {l}"
                    exp = st.expander(expander_label)

                    with exp:
                        try:
                            sr = tavily_client.search(query=l, search_depth="advanced", include_raw_content=True)
                            results = sr["results"]
                            sum_text = summarize_tavily_search(results)
                            st.write(sum_text)
                            mandatory_summaries += f"\n\n**Query {query_counter}:** {l}\n{sum_text}"
                        except Exception as e:
                            st.error(f"Error occurred during search: {str(e)}")
                    query_counter += 1

                st.session_state["all_research"] += mandatory_summaries
        except Exception as e:
            st.error(f"Error generating mandatory/topic-based queries: {str(e)}")

        # Update current step
        st.session_state["current_step"] = "outline_generation"

    # ------------------------------------------------------------------------------
    # Step 4: Generate and Display Outlines and Blogs Automatically
    # ------------------------------------------------------------------------------
    if st.session_state["current_step"] == "outline_generation" and st.session_state["all_research"]:
        st.header("4) Generating Blog Outlines and Corresponding Blogs")

        # Determine the number of iterations (e.g., 3 outlines and blogs)
        iterations = 3

        for i in range(iterations):
            outline_number = i + 1
            st.markdown(f"### Outline {outline_number}")

            # Prepare a list of previous outlines to ensure uniqueness
            previous_outlines = st.session_state["outlines_proposed"]

            try:
                with st.spinner(f"Generating Outline {outline_number}..."):
                    outline = generate_outline(
                        blog_topic=blog_topic,
                        target_audience=target_audience,
                        key_focus=key_focus,
                        article_length=article_length,
                        style_tone=style_tone,
                        keywords=keywords,
                        cta_text=cta_text,
                        additional_blog_insights=additional_blog_insights,
                        company_desc=st.session_state["company_info"],
                        product_desc=st.session_state["product_info"],
                        mandatory_summaries=st.session_state["all_research"],
                        previous_outlines=previous_outlines  # Pass previous outlines for uniqueness
                    )
                    st.success(f"**Proposed Outline {outline_number}:**")
                    st.write(outline)
                    st.session_state["outlines_proposed"].append(outline)
            except Exception as e:
                st.error(f"Error generating outline {outline_number}: {str(e)}")
                continue

            # Generate corresponding blog based on the current outline
            st.markdown(f"### Blog {outline_number} (Based on Outline {outline_number})")

            try:
                with st.spinner(f"Generating Blog {outline_number}..."):
                    blog = generate_blog(
                        chosen_outline=outline,
                        blog_topic=blog_topic,
                        target_audience=target_audience,
                        key_focus=key_focus,
                        article_length=article_length,
                        style_tone=style_tone,
                        keywords=keywords,
                        cta_text=cta_text,
                        additional_blog_insights=additional_blog_insights,
                        company_desc=st.session_state["company_info"],
                        product_desc=st.session_state["product_info"],
                        mandatory_summaries=st.session_state["all_research"]
                    )
                    st.success(f"**Final Blog Draft {outline_number}:**")
                    st.markdown(blog)  # Render Markdown
                    st.session_state["blogs_generated"].append(blog)
            except Exception as e:
                st.error(f"Error generating blog {outline_number}: {str(e)}")
                continue

        st.session_state["current_step"] = "completed"

    # ------------------------------------------------------------------------------
    # Step 6: Display All Generated Outlines and Blogs
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    # Initial Instructions
    # ------------------------------------------------------------------------------
    if st.session_state["current_step"] == "input" and not st.session_state["final_blog"]:
        st.write("Fill in the sidebar fields and click 'Start Content Generation Flow' to begin.")
        st.markdown("""
        ### Workflow Overview
        1. **Input Details:** Provide blog-related inputs via the sidebar.
        2. **Company & Product Info:** Optionally extract and summarize information from provided URLs.
        3. **Research Queries:** Automatically generate and summarize research queries.
        4. **Generate Outlines & Blogs:** Automatically create distinct outlines and corresponding blog posts with integrated research and citations.
        5. **Review Outputs:** View all generated content in an organized manner.
        """)

if __name__ == "__main__":
    main()
