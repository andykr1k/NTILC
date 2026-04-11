import { readFile } from "node:fs/promises";
import path from "node:path";
import type { ModelsRegistry, RegistryManifest, ToolRegistry } from "@/lib/types";

const generatedDir = path.resolve(process.cwd(), "..", "registry", "generated");

async function readJson<T>(filename: string): Promise<T> {
  const filePath = path.join(generatedDir, filename);
  const raw = await readFile(filePath, "utf8");
  return JSON.parse(raw) as T;
}

export function getToolRegistry() {
  return readJson<ToolRegistry>("tools.json");
}

export function getModelsRegistry() {
  return readJson<ModelsRegistry>("models.json");
}

export function getRegistryManifest() {
  return readJson<RegistryManifest>("registry_manifest.json");
}
