import * as fs from 'fs/promises';
import * as path from 'path';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { ChromaClient } from 'chromadb-client';
import { pipeline } from '@xenova/transformers';

let pdf: (buffer: Buffer | ArrayBuffer, options?: object) => Promise<{ text: string, numpages: number }> | undefined;
try {
  pdf = require('pdf-parse');
} catch (e) {
  console.error("CRITICAL ERROR: Failed to load 'pdf-parse'. Ensure it is installed correctly.", e);
  pdf = async () => { throw new Error("PDF parsing dependency 'pdf-parse' failed to load."); };
}

const DOCUMENTS_DIR = path.join(process.cwd(), 'documents');
const COLLECTION_NAME = 'kummatty_policies_xenova';
const CHROMA_HOST = process.env.CHROMA_HOST || 'http://localhost:8000';

const XENOVA_EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2';

const chromaClient = new ChromaClient({ path: CHROMA_HOST });

let extractor: any = null;

async function initializeExtractor() {
    if (extractor === null) {
        console.log(`Initializing Xenova pipeline with model: ${XENOVA_EMBEDDING_MODEL}...`);
        extractor = await pipeline('feature-extraction', XENOVA_EMBEDDING_MODEL);
        console.log("Xenova pipeline initialized successfully.");
    }
    return extractor;
}

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

async function extractTextFromPdf(filePath: string): Promise<string> {
  const dataBuffer = await fs.readFile(filePath);

  if (typeof pdf !== 'function') {
      throw new Error("PDF parsing dependency 'pdf-parse' is not available or failed to load.");
  }

  try {
    const data = await pdf(dataBuffer);
    return data.text.trim();
  } catch (parseError: any) {
    console.error(`Error parsing PDF file ${filePath}:`, parseError.message);
    throw new Error(`PDF Parsing failed for ${path.basename(filePath)}: ${parseError.message}`);
  }
}

/**
 * @param allChunks An array of text chunks.
 * @returns An array of embedding vectors (number[][]).
 */
async function generateEmbeddingsXenova(allChunks: string[]): Promise<number[][]> {
    const extractor = await initializeExtractor();
    console.log(`Generating embeddings for ${allChunks.length} chunks using Xenova/transformers...`);

    const embeddingsTensor = await extractor(allChunks, { pooling: 'mean', normalize: true });
    
    const embeddingsArray: number[][] = [];
    
    for (let i = 0; i < allChunks.length; i++) {
        const vector = embeddingsTensor[i].data;
        embeddingsArray.push(Array.from(vector));
    }
    
    console.log('All embeddings generated successfully by Xenova.');
    return embeddingsArray;
}

export async function ingestDocumentsXenova(): Promise<{ success: boolean; count: number; error?: string }> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
    separators: ["\n\n", "\n", " ", ""],
  });

  try {
    const collection = await getOrCreateCollection();
    
    const files = (await fs.readdir(DOCUMENTS_DIR)).filter(file => file.endsWith('.pdf'));
    
    if (files.length === 0) {
      return { success: false, count: 0, error: `No PDF files found in the directory: ${DOCUMENTS_DIR}` };
    }
    
    let chunkCount = 0;
    
    for (const fileName of files) {
      const filePath = path.join(DOCUMENTS_DIR, fileName);
      console.log(`\n--- Processing file: ${fileName} ---`);
      
      let fullText = '';
      try {
        fullText = await extractTextFromPdf(filePath);
      } catch (e: any) {
        console.warn(`[SKIPPING] Failed to parse ${fileName}: ${e.message}`);
        continue;
      }
      
      if (fullText.length === 0) {
        console.warn(`[SKIPPING] File ${fileName} could not be parsed or contains no text. Check if it's a scanned PDF.`);
        continue; 
      }
      
      const chunks = await splitter.splitText(fullText);

      if (chunks.length === 0) {
        console.warn(`[SKIPPING] File ${fileName} yielded text, but no chunks were created after splitting. Skipping ingestion.`);
        continue;
      }
      
      const embeddings = await generateEmbeddingsXenova(chunks);
      
      const ids: string[] = [];
      const metadatas: { source: string, chunk_index: number }[] = [];
      
      for (let i = 0; i < chunks.length; i++) {
        ids.push(`${fileName}-${i}`);
        metadatas.push({ 
            source: fileName, 
            chunk_index: i 
        });
      }
      
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
    console.error("XENOVA INGESTION FAILED:", e.message);
    return { success: false, count: 0, error: e.message };
  } finally {
      if (extractor) {
          extractor = null;
      }
  }
}