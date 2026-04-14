import type { ModelsRegistry, RegistryManifest, ToolRegistry } from "@/lib/types";
import modelsRegistry from "@/data/registry/models.json";
import registryManifest from "@/data/registry/registry_manifest.json";
import toolRegistry from "@/data/registry/tools.json";

export function getToolRegistry() {
  return Promise.resolve(toolRegistry as unknown as ToolRegistry);
}

export function getModelsRegistry() {
  return Promise.resolve(modelsRegistry as unknown as ModelsRegistry);
}

export function getRegistryManifest() {
  return Promise.resolve(registryManifest as unknown as RegistryManifest);
}
