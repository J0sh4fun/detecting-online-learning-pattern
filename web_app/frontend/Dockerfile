FROM node:20-alpine

WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy source code
COPY . .

# Run dev server exposing to host
CMD ["npm", "run", "dev", "--", "--host"]
