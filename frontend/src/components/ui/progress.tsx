"use client";

import * as React from "react";
import * as ProgressPrimitive from "@radix-ui/react-progress";

import { cn } from "./utils";

function Progress({
  className,
  value,
  indicatorClassName,
  ...props
}: React.ComponentProps<typeof ProgressPrimitive.Root> & { indicatorClassName?: string }) {
  // Ensure value is between 0 and 100
  const progressValue = Math.min(100, Math.max(0, value || 0));
  
  return (
    <ProgressPrimitive.Root
      data-slot="progress"
      value={progressValue}
      className={cn(
        "relative w-full overflow-hidden rounded-full bg-gray-200",
        className,
      )}
      style={{ height: '8px' }}
      {...props}
    >
      <ProgressPrimitive.Indicator
        data-slot="progress-indicator"
        className={cn(
          "h-full transition-all duration-300 ease-in-out",
          indicatorClassName || "bg-primary"
        )}
        style={{ width: `${progressValue}%` }}
      />
    </ProgressPrimitive.Root>
  );
}

export { Progress };
