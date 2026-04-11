import { NextResponse } from "next/server";
import { getRegistryManifest } from "@/lib/registry";

export async function GET() {
  return NextResponse.json(await getRegistryManifest());
}
