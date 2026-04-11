export type ParameterSpec = {
  type: string;
  description?: string;
  default?: string | number | boolean | null;
};

export type ToolParameters = {
  type: "object";
  properties: Record<string, ParameterSpec>;
  required: string[];
};

export type Category = {
  id: string;
  name: string;
  summary: string;
};

export type Tool = {
  id: string;
  display_name: string;
  description: string;
  interface_type: string;
  source_repo: string;
  homepage: string;
  license: string;
  maintainers: string[];
  parent_category: string;
  tags: string[];
  parameters: ToolParameters;
  example_count: number;
  has_tests: boolean;
  registry_path: string;
};

export type ToolRegistry = {
  version: number;
  generated_at: string;
  categories: Category[];
  tools: Tool[];
};

export type ModelRelease = {
  id: string;
  title: string;
  architecture: "normal" | "hierarchical";
  loss: "prototype_ce" | "contrastive" | "circle";
  encoder: string;
  embedding_dim: number;
  dataset_version: string;
  status: "planned" | "published" | "deprecated";
  published_at: string;
  download_url: string;
  repository_url: string;
  sha256: string;
  notes: string;
  metrics: Record<string, string | number | boolean | null>;
};

export type ModelsRegistry = {
  version: number;
  generated_at: string;
  releases: ModelRelease[];
};

export type RegistryManifest = {
  version: number;
  generated_at: string;
  tool_count: number;
  example_count: number;
  category_count: number;
  model_release_count: number;
  published_model_count: number;
  categories: Array<{
    id: string;
    name: string;
    tool_count: number;
  }>;
  artifacts: {
    tools: string;
    models: string;
    hierarchy: string;
    dataset: string;
  };
};
