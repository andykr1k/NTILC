import { NextResponse } from "next/server";
import { getToolRegistry } from "@/lib/registry";

export async function GET() {
  return NextResponse.json(await getToolRegistry());
}
