import { ImageResponse } from "next/og";

type SocialImageOptions = {
  eyebrow: string;
  title: string;
  description: string;
  chips?: string[];
  accent?: string;
  size?: {
    width: number;
    height: number;
  };
};

export const ogSize = {
  width: 1200,
  height: 630,
} as const;

export const twitterSize = {
  width: 1200,
  height: 630,
} as const;

export const socialImageContentType = "image/png";

export function createSocialImage({
  eyebrow,
  title,
  description,
  chips = [],
  accent = "#ff6b2c",
  size = ogSize,
}: SocialImageOptions) {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          position: "relative",
          overflow: "hidden",
          background:
            "radial-gradient(circle at 20% 18%, rgba(130, 182, 255, 0.18), transparent 24%), radial-gradient(circle at 82% 14%, rgba(255, 107, 44, 0.12), transparent 18%), linear-gradient(180deg, #060b12 0%, #0b1420 56%, #101c2c 100%)",
          color: "#edf4ff",
          fontFamily: "Trebuchet MS, sans-serif",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px)",
            backgroundSize: "28px 28px",
            opacity: 0.32,
          }}
        />

        <div
          style={{
            position: "absolute",
            inset: 28,
            border: "1px solid rgba(130, 182, 255, 0.18)",
            background: "rgba(9, 18, 28, 0.76)",
          }}
        />

        <div
          style={{
            position: "absolute",
            right: -120,
            top: -90,
            display: "flex",
            fontFamily: "Impact, Haettenschweiler, sans-serif",
            fontSize: 180,
            lineHeight: 0.8,
            letterSpacing: "-0.08em",
            textTransform: "uppercase",
            color: "rgba(237, 244, 255, 0.05)",
          }}
        >
          Open
        </div>

        <div
          style={{
            position: "absolute",
            right: 82,
            top: 78,
            width: 280,
            height: 280,
            display: "flex",
            border: "1px solid rgba(130, 182, 255, 0.24)",
            background:
              "linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(180deg, rgba(7, 14, 23, 0.92), rgba(11, 20, 32, 0.82))",
            backgroundSize: "22px 22px, 22px 22px, auto",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              position: "absolute",
              inset: "18% auto auto 18%",
              width: 120,
              height: 120,
              borderRadius: 9999,
              border: "1px solid rgba(130, 182, 255, 0.28)",
              boxShadow:
                "0 0 0 22px rgba(130, 182, 255, 0.05), 0 0 0 50px rgba(130, 182, 255, 0.03)",
            }}
          />
          <div
            style={{
              position: "absolute",
              inset: "-40% 0 auto 0",
              height: "55%",
              background: "linear-gradient(180deg, transparent, rgba(130, 182, 255, 0.24), transparent)",
              transform: "rotate(2deg)",
            }}
          />
        </div>

        <div
          style={{
            position: "relative",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            width: "100%",
            height: "100%",
            padding: "60px 68px",
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 18, maxWidth: 780 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 18,
              }}
            >
              <div
                style={{
                  display: "flex",
                  width: 76,
                  height: 76,
                  alignItems: "center",
                  justifyContent: "center",
                  border: "1px solid rgba(130, 182, 255, 0.2)",
                  background: "rgba(130, 182, 255, 0.08)",
                  color: "#82b6ff",
                  fontSize: 30,
                  fontWeight: 700,
                  letterSpacing: "0.18em",
                }}
              >
                OE
              </div>
              <div
                style={{
                  display: "flex",
                  fontSize: 22,
                  letterSpacing: "0.24em",
                  textTransform: "uppercase",
                  color: "#82b6ff",
                  fontWeight: 700,
                  fontFamily: "JetBrains Mono, monospace",
                }}
              >
                {eyebrow}
              </div>
            </div>

            <div
              style={{
                display: "flex",
                fontFamily: "Impact, Haettenschweiler, sans-serif",
                fontSize: 72,
                lineHeight: 0.92,
                letterSpacing: "-0.06em",
                textTransform: "uppercase",
              }}
            >
              {title}
            </div>

            <div
              style={{
                display: "flex",
                maxWidth: 760,
                fontFamily: "Baskerville, Georgia, serif",
                fontSize: 28,
                lineHeight: 1.45,
                color: "#9caec4",
              }}
            >
              {description}
            </div>
          </div>

          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "flex-end",
              gap: 24,
            }}
          >
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", maxWidth: 760 }}>
              {chips.map((chip) => (
                <div
                  key={chip}
                  style={{
                    display: "flex",
                    padding: "12px 16px",
                    border: "1px solid rgba(130, 182, 255, 0.16)",
                    background: "rgba(14, 25, 38, 0.66)",
                    fontSize: 18,
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    color: "#edf4ff",
                    fontFamily: "JetBrains Mono, monospace",
                  }}
                >
                  {chip}
                </div>
              ))}
            </div>

            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-end",
                fontFamily: "JetBrains Mono, monospace",
                color: accent,
                fontSize: 18,
                letterSpacing: "0.12em",
                textTransform: "uppercase",
              }}
            >
              <div>Registry on GitHub</div>
              <div>Models on Hugging Face</div>
            </div>
          </div>
        </div>
      </div>
    ),
    size,
  );
}
