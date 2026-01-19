ARG USERIMAGE=project:tag
FROM ${USERIMAGE} AS base

# Install custom dependencies
# RUN apt-get update && apt-get install --no-install-recommends -y \
#     <YOUR_PACKAGE> \
#     && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry completions bash > /etc/bash_completion.d/poetry
RUN poetry config virtualenvs.create false 

# Install project dependencies
WORKDIR /workspace/automatic_assessment
COPY /project/automatic_assessment/pyproject.toml .
COPY /project/automatic_assessment/poetry.lock .
RUN poetry install --no-root --no-interaction

# Install the project
COPY /project/automatic_assessment .
RUN poetry install --only-root

# # Initialize PYTHONPATH to avoid undefined variable warning
# ENV PYTHONPATH=""
# ENV PYTHONPATH="/workspace/sirius-ml-env/sirius-mask-prediction-network:${PYTHONPATH}"

CMD ["/bin/bash"]
