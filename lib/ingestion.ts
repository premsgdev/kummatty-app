import * as fs from 'fs';
import * as path from 'path';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenAI } from '@google/genai';
import { ChromaClient } from 'chromadb-client';

// --- PDF-PARSE REPLACEMENT: Using pdf2json ---
// pdf2json provides a more stable interface for Node.js/Next.js environments.
const PDFParser = require("pdf2json");

// --- CONFIGURATION ---
const DOCUMENTS_DIR = path.join(process.cwd(), 'documents');
const COLLECTION_NAME = 'kummatty_policies';
const EMBEDDING_MODEL = 'models/embedding-001';
const CHROMA_HOST = process.env.CHROMA_HOST || 'http://localhost:8000';

// Initialize AI and ChromaDB clients
const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  throw new Error("GEMINI_API_KEY environment variable not set.");
}
const ai = new GoogleGenAI({ apiKey });
const chromaClient = new ChromaClient({ path: CHROMA_HOST });

// --- RAG UTILITY FUNCTIONS ---

/**
 * Ensures the ChromaDB collection exists and returns it.
 */
async function getOrCreateCollection() {
  console.log(`Attempting to connect to Chroma at: ${CHROMA_HOST}`);
  
  try {
    const collection = await chromaClient.getOrCreateCollection({ name: COLLECTION_NAME });
    console.log(`Successfully connected and retrieved/created collection: ${COLLECTION_NAME}`);
    return collection;
  } catch (error) {
    console.error("Error connecting to ChromaDB:", error);
    throw new Error("Failed to connect to ChromaDB. Ensure the service is running at " + CHROMA_HOST);
  }
}

/**
 * Parses a PDF file content into plain text using pdf2json.
 * @param filePath The path to the PDF file.
 * @returns The extracted text.
 */
async function extractTextFromPdf(filePath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    // pdf2json requires instantiation and event handling
    const pdfParser = new PDFParser(null, 1);
    
    // Success event listener
    pdfParser.on("pdfParser_dataReady", (pdfData: { Pages: { Texts: { R: { T: string }[] }[] }[] }) => {
      // Concatenate text from all pages
      let fullText = "";
      
      pdfData.Pages.forEach(page => {
        // pdf2json encodes text, we must decode and join
        const pageText = page.Texts
          .map(textBlock => textBlock.R)
          .flat()
          .map(textRun => decodeURIComponent(textRun.T))
          .join(' ');
        
        fullText += pageText + '\n\n';
      });

      resolve(fullText);
    });

    // Error event listener
    pdfParser.on("pdfParser_dataError", (errData: { parserError: string }) => {
      reject(new Error(`PDF parsing error: ${errData.parserError}`));
    });

    // Load the file buffer directly
    pdfParser.loadPDF(filePath);
  });
}

/**
 * Splits text into manageable chunks for RAG.
 * @param text The full document text.
 * @returns An array of text chunks.
 */
async function chunkText(text: string): Promise<string[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
    separators: ["\n\n", "\n", " ", ""],
  });
  const chunks = await splitter.splitText(text);
  console.log(`Split document into ${chunks.length} chunks.`);
  return chunks;
}

/**
 * Generates vector embeddings for a batch of text chunks.
 * @param chunks An array of text chunks.
 * @returns An array of embedding vectors.
 */
async function generateEmbeddings(chunks: string[]): Promise<number[][]> {
  console.log(`Generating embeddings for ${chunks.length} chunks using ${EMBEDDING_MODEL}...`);
  
  const response = await ai.models.embedContent({
    model: EMBEDDING_MODEL,
    contents: chunks.map(text => ({ parts: [{ text }] })),
  });

  if (!response.embeddings || response.embeddings.length === 0) {
    console.error("Embedding API Response:", response);
    throw new Error("Failed to generate embeddings. The API response did not contain the expected 'embeddings' data.");
  }

  const embeddings = response.embeddings.map(e => e.values as number[]);
  console.log('Embeddings generated successfully.');
  return embeddings;
}

/**
 * Main function to ingest documents from the local directory into ChromaDB.
 */
export async function ingestDocuments(): Promise<{ success: boolean; count: number; error?: string }> {
  try {
    const collection = await getOrCreateCollection();
    
    const files = fs.readdirSync(DOCUMENTS_DIR).filter(file => file.endsWith('.pdf'));
    if (files.length === 0) {
      return { success: false, count: 0, error: `No PDF files found in the directory: ${DOCUMENTS_DIR}` };
    }
    
    let chunkCount = 0;
    
    for (const fileName of files) {
      const filePath = path.join(DOCUMENTS_DIR, fileName);
      console.log(`\n--- Processing file: ${fileName} ---`);
      
      const fullText = await extractTextFromPdf(filePath);
      const chunks = await chunkText(fullText);
      const embeddings = await generateEmbeddings(chunks);
      
      const ids: string[] = [];
      const metadatas: { source: string, chunk_index: number }[] = [];
      
      for (let i = 0; i < chunks.length; i++) {
        ids.push(`${fileName}-${i}`);
        metadatas.push({ source: fileName, chunk_index: i });
      }
      
      // Store the chunks, embeddings, and metadata in ChromaDB
      await collection.upsert({
        ids: ids,
        embeddings: embeddings,
        documents: chunks,
        metadatas: metadatas
      });
      
      chunkCount += chunks.length;
      console.log(`File ${fileName} successfully ingested. Total chunks added/updated: ${chunkCount}`);
    }
    
    return { success: true, count: chunkCount };
    
  } catch (e: any) {
    console.error("INGESTION FAILED:", e.message);
    return { success: false, count: 0, error: e.message };
  }
}