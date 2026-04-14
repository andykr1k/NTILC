import { NextResponse } from "next/server";
import { getModelsRegistry } from "@/lib/registry";

export async function GET() {
  return NextResponse.json(await getModelsRegistry());
}
