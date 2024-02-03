from setuptools import setup

setup(
    name='R4B',
    version='0.1.0',    
    description='A wrapper around Langchain and pinecone to perform Retrieval Augmented Gneration. Include your OpenAI and Pinecone API keys, and start querying documents using LLMs!',
    url='https://github.com/amaygada/RAG4Babies',
    author='Amay Gada',
    author_email='amaygada@gmail.com',
    license='BSD 2-clause',
    packages=['R4B'],
    install_requires=['langchain==0.0.272',
                      'openai==0.27.9',
                      'numpy==1.25.2',
                      'pandas==2.1.0',
                      'python-dotenv==1.0.0',
                      'pinecone-client==2.2.2',
                      'tiktoken==0.5.1',
                      'pdfminer-six==20221105',
                      'sentence-transformers==2.2.2',
                      'beautifulsoup4',
                      'pinecone-text[splade]',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.10',
    ],
)